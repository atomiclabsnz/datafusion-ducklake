//! ADR-0037 slice 4 (in ferrion): a sorted `INSERT` and a sorted compaction
//! sort past the memory pool by spilling, under the session's pool and disk
//! manager, and are refused whole where they cannot.
//!
//! A writer-sorted `INSERT` (an expression key with `sort_on_insert` on, or the
//! option off while inlining is enabled) used to collect its input and sort a
//! second copy of it in memory. It now streams through DataFusion's spilling
//! sort. The reference every layout here is held to is that in-memory sort:
//! [`DuckLakeTableWriter::write_rows`] still makes it, over the same batches, so
//! "the same rows, order and files" is checked file by file, row by row,
//! against the code path the SQL `INSERT` took before.

#![cfg(feature = "write-sqlite")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, AsArray, Int64Array, RecordBatch, StringViewArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use datafusion::datasource::MemTable;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::{DiskManager, SpillFile, TempFileFactory};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};
use datafusion_ducklake::{DuckLakeWriteOptions, MergeOptions, RewriteOptions, WriteMode};
use object_store::local::LocalFileSystem;

use crate::inlining_abstraction_tests::{Lake, create_with, sqlite};
use crate::sort_expression_tests::raw;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;

/// The pool a spilling run is held to: a fraction of what it sorts, with room
/// for the writer beside the sort (an open file's dictionary column under
/// `PARQUET_2_0` holds about 1 MiB of encoder, so the session is charged about
/// 2 MiB from its first row group).
const SMALL_POOL: usize = 8 * MIB;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("k", DataType::Int64, true),
        // A table's VARCHAR, as the fork reads and writes it.
        Field::new("v", DataType::Utf8View, true),
    ]))
}

/// `rows` rows in batches of `per_batch` (an odd size, so the lengths the sort
/// is cut back to are not its own): `id` the row's arrival position, `k` a
/// scrambled key with many ties and a NULL every 13th row, `v` 40 characters
/// that do not repeat, so the rows are megabytes and a tie's order shows.
fn input(rows: i64, per_batch: i64) -> Vec<RecordBatch> {
    (0..rows)
        .step_by(per_batch as usize)
        .map(|start| {
            let ids = start..(start + per_batch).min(rows);
            let k: Int64Array = ids
                .clone()
                .map(|i| (i % 13 != 0).then_some((i * 7919) % 1009))
                .collect();
            let v: StringViewArray = ids
                .clone()
                .map(|i| Some(format!("{:040}", (i * 2654435761) % 1000000007)))
                .collect();
            RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from_iter_values(ids)), Arc::new(k), Arc::new(v)],
            )
            .unwrap()
        })
        .collect()
}

/// Row groups of 1,000 rows and a 64 KiB target, so a write rolls into many
/// files; `limit` rows inline.
fn options(limit: usize) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.max_row_group_rows = Some(1000);
    options.target_file_size = Some(64 * KIB);
    options.data_inlining_row_limit = Some(limit);
    // The compression a lake's INSERT writes with, its default, so that
    // `write_rows` (which otherwise writes uncompressed) writes the same files
    // an INSERT wrote before the sort streamed.
    options.compression = Some(datafusion::parquet::basic::Compression::SNAPPY);
    options
}

type Key<'a> = (&'a str, SortDirection, NullOrder);

/// `main.<name>(id, k, v)` sorted by `keys` (expressions as DuckDB stores
/// them), with `sort_on_insert` stored as given: its table id.
async fn sorted_table(lake: &Lake, name: &str, keys: &[Key<'_>], sort_on_insert: bool) -> i64 {
    create_with(
        lake,
        name,
        &[("id", DataType::Int64), ("k", DataType::Int64), ("v", DataType::Utf8)],
    )
    .unwrap();
    let table_id = lake.table_id(name);
    let fields: Vec<SortField> = keys
        .iter()
        .enumerate()
        .map(|(index, (_, direction, null_order))| SortField {
            sort_key_index: index as i32,
            // A bare column, which `set_sort_spec` takes; the key's own
            // expression is written over it below, as DuckDB stores it.
            expression: "id".to_string(),
            dialect: "duckdb".to_string(),
            direction: *direction,
            null_order: *null_order,
        })
        .collect();
    lake.writer.set_sort_spec(table_id, &fields).unwrap();
    for (index, (expression, _, _)) in keys.iter().enumerate() {
        raw(
            lake,
            &format!(
                "UPDATE ducklake_sort_expression SET expression = '{}' \
                 WHERE table_id = {table_id} AND sort_key_index = {index}",
                expression.replace('\'', "''")
            ),
        )
        .await;
    }
    lake.writer
        .set_table_setting(
            table_id,
            "sort_on_insert",
            if sort_on_insert {
                "true"
            } else {
                "false"
            },
        )
        .unwrap();
    table_id
}

/// A session over `lake` (writing under `options`) on `runtime`, one
/// partition as a tab runs, with `src` registered as `batches`.
fn session(
    lake: &Lake,
    options: &DuckLakeWriteOptions,
    runtime: RuntimeEnvBuilder,
    batches: &[RecordBatch],
) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .with_sort_spill_reservation_bytes(256 * KIB);
    let ctx = SessionContext::new_with_config_rt(config, runtime.build_arc().unwrap());
    ctx.register_catalog(
        "lake",
        Arc::new(lake.catalog().with_write_options(options.clone())),
    );
    ctx.register_table(
        "src",
        Arc::new(MemTable::try_new(schema(), vec![batches.to_vec()]).unwrap()),
    )
    .unwrap();
    ctx
}

/// A runtime whose pool is `pool` bytes, spilling to the OS temp directory
/// through `factory` when given, or not at all.
fn runtime(pool: usize, factory: Option<Arc<CountingFactory>>) -> RuntimeEnvBuilder {
    let disk = match factory {
        Some(factory) => DiskManagerBuilder::default().with_mode(DiskManagerMode::Custom(factory)),
        None => DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
    };
    RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(pool)))
        .with_disk_manager_builder(disk)
}

/// Spill files through a disk manager on the OS temp directory, counted, and
/// refused past `limit` bytes of them at once (a spill budget).
#[derive(Debug)]
struct CountingFactory {
    inner: Arc<DiskManager>,
    created: AtomicUsize,
}

impl CountingFactory {
    fn new(limit: Option<u64>) -> Arc<Self> {
        let mut builder = DiskManagerBuilder::default().with_mode(DiskManagerMode::OsTmpDirectory);
        if let Some(limit) = limit {
            builder = builder.with_max_temp_directory_size(limit);
        }
        Arc::new(Self {
            inner: Arc::new(builder.build().unwrap()),
            created: AtomicUsize::new(0),
        })
    }

    fn created(&self) -> usize {
        self.created.load(Ordering::SeqCst)
    }
}

impl TempFileFactory for CountingFactory {
    fn create_temp_file(&self, description: &str) -> datafusion::error::Result<Arc<dyn SpillFile>> {
        self.created.fetch_add(1, Ordering::SeqCst);
        self.inner.create_tmp_file(description)
    }
}

/// Run `sql` in `ctx`: the plan it executed, or its error.
async fn run(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>, String> {
    let plan = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .create_physical_plan()
        .await
        .map_err(|e| e.to_string())?;
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .map_err(|e| e.to_string())?;
    Ok(plan)
}

/// Spills `plan` recorded: `(the insert's own, every other node's)`.
fn spills(plan: &Arc<dyn ExecutionPlan>) -> (usize, usize) {
    let own = |plan: &Arc<dyn ExecutionPlan>| {
        plan.metrics()
            .and_then(|metrics| metrics.spill_count())
            .unwrap_or(0)
    };
    fn others(
        plan: &Arc<dyn ExecutionPlan>,
        own: &dyn Fn(&Arc<dyn ExecutionPlan>) -> usize,
    ) -> usize {
        plan.children()
            .into_iter()
            .map(|child| own(child) + others(child, own))
            .sum()
    }
    (own(plan), others(plan, &own))
}

/// One data file: `(record count, bytes, its rows in physical order)`.
type File = (i64, i64, Vec<(i64, Option<i64>, String)>);

/// `main.<table>`'s data files in data file id order, each read from disk.
fn layout(lake: &Lake, table: &str) -> Vec<File> {
    fn find(dir: &Path, name: &std::ffi::OsStr) -> Option<PathBuf> {
        for entry in std::fs::read_dir(dir).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                if let Some(found) = find(&path, name) {
                    return Some(found);
                }
            } else if path.file_name() == Some(name) {
                return Some(path);
            }
        }
        None
    }
    let mut files = lake
        .provider
        .get_table_files_for_select(lake.table_id(table), lake.head())
        .unwrap();
    files.sort_by_key(|file| file.data_file_id);
    files
        .iter()
        .map(|file| {
            let name = Path::new(&file.file.path).file_name().unwrap().to_owned();
            let path = find(&lake.data, &name).expect("data file on disk");
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(path).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap();
            let mut rows = Vec::new();
            for batch in reader {
                let batch = batch.unwrap();
                let ids = batch.column(0).as_primitive::<Int64Type>();
                let ks = batch.column(1).as_primitive::<Int64Type>();
                let vs = arrow::compute::cast(batch.column(2), &DataType::Utf8).unwrap();
                let vs = vs.as_string::<i32>();
                for row in 0..batch.num_rows() {
                    rows.push((
                        ids.value(row),
                        ks.is_valid(row).then(|| ks.value(row)),
                        vs.value(row).to_string(),
                    ));
                }
            }
            (rows.len() as i64, file.file.file_size_bytes, rows)
        })
        .collect()
}

/// Every file under `main.<table>`'s directory.
fn on_disk(lake: &Lake, table: &str) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    out.push(path);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(&lake.data.join("main").join(table), &mut out);
    out.sort();
    out
}

/// `main.<table>` written by the in-memory sort, as a SQL `INSERT` wrote it
/// before: [`DuckLakeTableWriter::write_rows`] over `batches`.
async fn in_memory(
    lake: &Lake,
    table: &str,
    options: &DuckLakeWriteOptions,
    batches: &[RecordBatch],
) {
    lake.table_writer()
        .with_options(options)
        .write_rows("main", table, &schema(), WriteMode::Append, batches)
        .await
        .unwrap();
}

/// A writer-sorted INSERT past a small pool spills (the insert's own sort
/// reports it) and lays the rows out exactly as the collected write did before
/// slice 4 (`write_rows`, its sort `sort_insert_rows`): the same files, each
/// with the same rows in the same order (ties in arrival order) and the same
/// bytes. Under an unbounded pool it spills nothing and lays them out the same
/// way.
/// Each key shape that leaves the sort to the writer: an expression key with
/// `sort_on_insert` on (one key, then two, in both directions and null
/// orders), and a bare column with the option off while inlining is enabled.
#[tokio::test(flavor = "multi_thread")]
async fn a_writer_sorted_insert_spills_and_lays_out_as_the_collected_write() {
    use NullOrder::*;
    use SortDirection::*;
    let batches = input(120_000, 997);
    let cases: [(&[Key<'_>], bool, usize); 4] = [
        (&[("(k % 7)", Asc, NullsLast)], true, 0),
        (
            &[("(k % 7)", Desc, NullsFirst), ("abs((k - 500))", Asc, NullsLast)],
            true,
            0,
        ),
        (&[("(k + 1)", Desc, NullsLast)], true, 10),
        (&[("k", Desc, NullsFirst)], false, 10),
    ];
    for (keys, sort_on_insert, limit) in cases {
        let lake = sqlite()(None).await;
        let options = options(limit);
        for table in ["a", "b", "c"] {
            sorted_table(&lake, table, keys, sort_on_insert).await;
        }
        // The reference: the collected write and its sort (`sort_insert_rows`),
        // what an INSERT wrote before its sort streamed. Below, the same files,
        // each with the same rows in the same order (ties in arrival order) and
        // the same bytes.
        in_memory(&lake, "a", &options, &batches).await;
        let expected = layout(&lake, "a");
        assert!(expected.len() > 5, "{keys:?}: rolls into many files");
        // A file rolls between batches, and the sorted rows are cut back to
        // the lengths they arrived in (997, the sort's own being 8,192): every
        // file but the last holds a whole number of them, as before.
        assert!(
            expected[..expected.len() - 1]
                .iter()
                .all(|file| file.0 % 997 == 0),
            "{keys:?}: files roll at the input's batch boundaries"
        );

        // The same INSERT under an unbounded pool, sorted in memory.
        let ctx = session(&lake, &options, RuntimeEnvBuilder::new(), &batches);
        let plan = run(&ctx, "INSERT INTO lake.main.c SELECT id, k, v FROM src")
            .await
            .unwrap_or_else(|e| panic!("{keys:?}: {e}"));
        assert_eq!(spills(&plan), (0, 0), "{keys:?}: in memory");
        assert!(
            layout(&lake, "c") == expected,
            "{keys:?}: in memory, laid out as the collected write"
        );

        let factory = CountingFactory::new(None);
        let ctx = session(
            &lake,
            &options,
            runtime(SMALL_POOL, Some(Arc::clone(&factory))),
            &batches,
        );
        let plan = run(&ctx, "INSERT INTO lake.main.b SELECT id, k, v FROM src")
            .await
            .unwrap_or_else(|e| panic!("{keys:?}: {e}"));
        let (own, others) = spills(&plan);
        assert!(
            own > 0,
            "{keys:?}: the insert's sort spilled ({own}, the plan's {others})"
        );
        assert_eq!(others, 0, "{keys:?}: the plan sorted nothing");
        assert!(factory.created() > 0, "{keys:?}: spill files were made");
        assert!(
            layout(&lake, "b") == expected,
            "{keys:?}: spilled, laid out as the collected write"
        );
    }
}

/// A key that fails over the rows (`CAST` of a string that is no number, on
/// the last rows only) leaves every row in arrival order, as the in-memory
/// sort leaves it, spilled or not. Two sorts hold memory at once here (the
/// key's, merging, and the position's, filling), so the pool is twice the
/// others' and the rows twice as many.
#[tokio::test(flavor = "multi_thread")]
async fn a_failing_key_writes_the_rows_in_arrival_order() {
    let mut batches = input(240_000, 997);
    // The last row's `v` is no number: `CAST(v AS BIGINT)` fails there.
    let last = batches.pop().unwrap();
    let n = last.num_rows();
    let v: StringViewArray = (0..n)
        .map(|row| {
            Some(if row == n - 1 {
                "x".to_string()
            } else {
                last.column(2).as_string_view().value(row).to_string()
            })
        })
        .collect();
    batches.push(
        RecordBatch::try_new(
            schema(),
            vec![Arc::clone(last.column(0)), Arc::clone(last.column(1)), Arc::new(v)],
        )
        .unwrap(),
    );
    let keys: &[Key<'_>] = &[(
        "CAST(v AS BIGINT)",
        SortDirection::Asc,
        NullOrder::NullsLast,
    )];
    let lake = sqlite()(None).await;
    let options = options(0);
    for table in ["a", "b", "c"] {
        sorted_table(&lake, table, keys, true).await;
    }
    let ids = |files: &[File]| -> Vec<i64> {
        files.iter().flat_map(|f| f.2.iter().map(|r| r.0)).collect()
    };
    in_memory(&lake, "a", &options, &batches).await;
    let expected = layout(&lake, "a");
    assert_eq!(
        ids(&expected),
        (0..240_000).collect::<Vec<_>>(),
        "the collected sort keeps arrival order"
    );
    let ctx = session(&lake, &options, RuntimeEnvBuilder::new(), &batches);
    run(&ctx, "INSERT INTO lake.main.c SELECT id, k, v FROM src")
        .await
        .unwrap();
    assert!(
        layout(&lake, "c") == expected,
        "in memory, laid out as the collected write"
    );
    let factory = CountingFactory::new(None);
    let ctx = session(
        &lake,
        &options,
        runtime(2 * SMALL_POOL, Some(Arc::clone(&factory))),
        &batches,
    );
    let plan = run(&ctx, "INSERT INTO lake.main.b SELECT id, k, v FROM src")
        .await
        .unwrap();
    assert!(spills(&plan).0 > 0, "the insert's sorts spilled");
    assert!(
        layout(&lake, "b") == expected,
        "spilled, laid out as the collected write"
    );
}

/// The catalog's snapshots, data files and inlined rows of `main.<table>`.
fn committed(lake: &Lake, table: &str) -> (i64, usize, usize) {
    (
        lake.head(),
        lake.data_files(table),
        lake.inlined_rows(table),
    )
}

/// A writer-sorted INSERT that fails mid-sort leaves nothing: no catalog row
/// and no file. Refused at the pool with no disk manager to spill to; past its
/// spill budget; and by its input, failing on its last rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_insert_refused_mid_sort_leaves_nothing() {
    let batches = input(120_000, 997);
    let keys: &[Key<'_>] = &[("(k % 7)", SortDirection::Asc, NullOrder::NullsLast)];
    let insert = "INSERT INTO lake.main.b SELECT id, k, v FROM src";
    let failing = "INSERT INTO lake.main.b SELECT id, \
                   CASE WHEN id = 119990 THEN CAST(concat('x', v) AS BIGINT) ELSE k END, v FROM src";
    let cases: [(&str, RuntimeEnvBuilder, &str, &str); 3] = [
        (
            "the pool, no disk manager",
            runtime(SMALL_POOL, None),
            insert,
            "Resources exhausted",
        ),
        (
            "the spill budget",
            runtime(
                SMALL_POOL,
                Some(CountingFactory::new(Some(64 * KIB as u64))),
            ),
            insert,
            "exceeded",
        ),
        (
            "the input",
            runtime(SMALL_POOL, Some(CountingFactory::new(None))),
            failing,
            "Cannot cast",
        ),
    ];
    for (what, runtime, sql, says) in cases {
        let lake = sqlite()(None).await;
        sorted_table(&lake, "b", keys, true).await;
        let before = committed(&lake, "b");
        let ctx = session(&lake, &options(0), runtime, &batches);
        let error = run(&ctx, sql).await.expect_err(what);
        assert!(error.contains(says), "{what}: {error}");
        assert_eq!(committed(&lake, "b"), before, "{what}: nothing committed");
        assert_eq!(
            on_disk(&lake, "b"),
            Vec::<PathBuf>::new(),
            "{what}: no file left"
        );
    }
}

/// A local store whose uploads fail once `allowed` have started.
#[derive(Debug)]
struct FailingStore {
    inner: LocalFileSystem,
    allowed: usize,
    started: AtomicUsize,
}

impl std::fmt::Display for FailingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingStore")
    }
}

impl FailingStore {
    fn admit(&self) -> object_store::Result<()> {
        if self.started.fetch_add(1, Ordering::SeqCst) < self.allowed {
            Ok(())
        } else {
            Err(object_store::Error::Generic {
                store: "FailingStore",
                source: "injected upload failure".into(),
            })
        }
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FailingStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.admit()?;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.admit()?;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// A sorted INSERT whose write fails after the sort (the fourth file's
/// upload) aborts its session: the files it wrote before are removed and no
/// catalog row is left.
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_insert_whose_write_fails_after_the_sort_leaves_nothing() {
    let batches = input(120_000, 997);
    let keys: &[Key<'_>] = &[("(k % 7)", SortDirection::Asc, NullOrder::NullsLast)];
    let lake = sqlite()(None).await;
    sorted_table(&lake, "b", keys, true).await;
    let before = committed(&lake, "b");
    let factory = CountingFactory::new(None);
    let ctx = session(
        &lake,
        &options(0),
        runtime(SMALL_POOL, Some(Arc::clone(&factory))),
        &batches,
    );
    ctx.runtime_env().register_object_store(
        datafusion::execution::object_store::ObjectStoreUrl::local_filesystem().as_ref(),
        Arc::new(FailingStore {
            inner: LocalFileSystem::new(),
            allowed: 3,
            started: AtomicUsize::new(0),
        }),
    );
    let error = run(&ctx, "INSERT INTO lake.main.b SELECT id, k, v FROM src")
        .await
        .expect_err("the fourth upload fails");
    assert!(error.contains("injected upload failure"), "{error}");
    assert!(factory.created() > 0, "it failed after the sort spilled");
    assert_eq!(committed(&lake, "b"), before, "nothing committed");
    assert_eq!(on_disk(&lake, "b"), Vec::<PathBuf>::new(), "no file left");
}

/// A sort the plan makes (a bare column with `sort_on_insert` on) is
/// DataFusion's `SortExec` in the insert's plan: past a small pool it spills,
/// and every row lands, in key order. Its ties are in the order DataFusion's
/// sort leaves them (it is not stable), as before slice 4: this sort did not
/// change, only the session's room beside it.
#[tokio::test(flavor = "multi_thread")]
async fn a_plan_sorted_insert_spills() {
    let batches = input(120_000, 997);
    let lake = sqlite()(None).await;
    sorted_table(
        &lake,
        "b",
        &[("k", SortDirection::Asc, NullOrder::NullsLast)],
        true,
    )
    .await;
    let ctx = session(
        &lake,
        &options(0),
        runtime(SMALL_POOL, Some(CountingFactory::new(None))),
        &batches,
    );
    let plan = run(&ctx, "INSERT INTO lake.main.b SELECT id, k, v FROM src")
        .await
        .unwrap();
    let (own, others) = spills(&plan);
    assert_eq!(own, 0, "the writer sorted nothing");
    assert!(others > 0, "the plan's sort spilled");
    let files = layout(&lake, "b");
    let mut ids: Vec<i64> = files.iter().flat_map(|f| f.2.iter().map(|r| r.0)).collect();
    let keys: Vec<Option<i64>> = files
        .into_iter()
        .flat_map(|file| file.2.into_iter().map(|row| row.1))
        .collect();
    assert_eq!(keys.len(), 120_000);
    let rank = |k: &Option<i64>| k.map_or(i64::MAX, |k| k);
    assert!(
        keys.windows(2).all(|w| rank(&w[0]) <= rank(&w[1])),
        "in `k` order, NULLs last"
    );
    ids.sort_unstable();
    assert_eq!(ids, (0..120_000).collect::<Vec<_>>(), "every row, once");
}

/// `main.t` sorted by `(k % 7)` (compaction sorts by it), written unsorted as
/// four files: two of distinct strings, then two whose rows repeat 7 values in
/// runs of 100, so a small file holds megabytes of rows. A row of the last two
/// is all its key's (`k` is its own `k % 7`): rows that tie on it are equal, so
/// no tie rule can tell two layouts of them apart.
async fn compaction_lake() -> Lake {
    let lake = sqlite()(None).await;
    sorted_table(
        &lake,
        "t",
        &[("(k % 7)", SortDirection::Asc, NullOrder::NullsLast)],
        false,
    )
    .await;
    let mut options = options(0);
    options.target_file_size = None;
    // A row group a file, so the two big files are small on disk (their
    // dictionary pages are written once, not once per 1,000 rows).
    options.max_row_group_rows = None;
    let small = input(14_000, 997);
    let repeated = |b: i64| -> RecordBatch {
        let ks = (0..5000i64).map(|i| (i / 100 + b) % 7);
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(ks.clone().map(|_| 0))),
                Arc::new(ks.clone().map(Some).collect::<Int64Array>()),
                Arc::new(StringViewArray::from_iter_values(
                    ks.map(|k| format!("{}{k}", "y".repeat(100))),
                )),
            ],
        )
        .unwrap()
    };
    let big = |from: i64| -> Vec<RecordBatch> { (from..from + 40).map(repeated).collect() };
    for batches in [small[..7].to_vec(), small[7..].to_vec(), big(0), big(40)] {
        lake.table_writer()
            .with_options(&options)
            .write_rows("main", "t", &schema(), WriteMode::Append, &batches)
            .await
            .unwrap();
    }
    assert_eq!(lake.data_files("t"), 4);
    lake
}

/// Merge every file of `lake`'s `main.t` into bins of about `target` bytes, on
/// `runtime`.
async fn merge(lake: &Lake, runtime: RuntimeEnvBuilder, target: u64) -> Result<usize, String> {
    let ctx = session(lake, &options(0), runtime, &input(1, 1));
    let table = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
        .expect("a DuckLakeTable");
    table
        .merge_adjacent_files(
            &ctx.state(),
            MergeOptions {
                target_file_size: target,
                max_merged_files: 100,
                min_file_size: 0,
            },
        )
        .await
        .map(|merged| merged.files_created)
        .map_err(|e| e.to_string())
}

/// A sorted compaction past a small pool spills and writes the files the same
/// compaction in memory does (the same rows each, in key order; ties as
/// DataFusion's sort leaves them); with nothing to spill to it is refused, not
/// written unsorted, and the bin it had already merged is removed: the
/// catalog and the files are as they were.
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_compaction_spills_or_is_refused_whole() {
    let lake = compaction_lake().await;
    let sizes: Vec<i64> = layout(&lake, "t").iter().map(|file| file.1).collect();
    // The two small files reach the target together; the two big ones, small
    // on disk, are the second bin.
    let target = (sizes[0] + sizes[1]) as u64;
    assert!(sizes[2] + sizes[3] < target as i64, "{sizes:?}");

    let before = (committed(&lake, "t"), on_disk(&lake, "t"));
    let refused = merge(&lake, runtime(SMALL_POOL, None), target)
        .await
        .expect_err("the second bin's sort does not fit the pool");
    assert!(refused.contains("Resources exhausted"), "{refused}");
    assert_eq!(
        (committed(&lake, "t"), on_disk(&lake, "t")),
        before,
        "nothing left"
    );

    let reference = compaction_lake().await;
    assert_eq!(
        merge(&reference, RuntimeEnvBuilder::new(), target).await,
        Ok(2)
    );
    let factory = CountingFactory::new(None);
    assert_eq!(
        merge(
            &lake,
            runtime(SMALL_POOL, Some(Arc::clone(&factory))),
            target
        )
        .await,
        Ok(2)
    );
    assert!(factory.created() > 0, "the compaction's sort spilled");
    let rows = |lake: &Lake| -> Vec<Vec<(Option<i64>, String)>> {
        layout(lake, "t")
            .into_iter()
            .map(|file| file.2.into_iter().map(|row| (row.1, row.2)).collect())
            .collect()
    };
    let spilled = rows(&lake);
    // The same files, each with the same rows. Not in the same order among the
    // rows that tie on `(k % 7)`: compaction's sort has no tie-break (before
    // slice 4 neither), and DataFusion's sort orders ties as its runs fall,
    // which a pool changes. The big bin's ties are equal rows (the fixture).
    let as_set = |files: &[Vec<(Option<i64>, String)>]| -> Vec<Vec<(Option<i64>, String)>> {
        files
            .iter()
            .map(|file| {
                let mut file = file.clone();
                file.sort();
                file
            })
            .collect()
    };
    assert_eq!(
        as_set(&spilled),
        as_set(&rows(&reference)),
        "the same files, the same rows"
    );
    for file in &spilled {
        let key = |row: &(Option<i64>, String)| row.0.map_or(i64::MAX, |k| k % 7);
        assert!(
            file.windows(2).all(|w| key(&w[0]) <= key(&w[1])),
            "in `(k % 7)` order"
        );
    }
}

/// Rewrite every file of `lake`'s `main.t` that has deletions, on `runtime`.
async fn rewrite(lake: &Lake, runtime: RuntimeEnvBuilder) -> Result<usize, String> {
    let ctx = session(lake, &options(0), runtime, &input(1, 1));
    let table = ctx
        .catalog("lake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap();
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<datafusion_ducklake::DuckLakeTable>()
        .expect("a DuckLakeTable");
    table
        .rewrite_data_files(
            &ctx.state(),
            RewriteOptions {
                delete_threshold: 0.0,
                data_file_ids: None,
            },
        )
        .await
        .map(|rewritten| rewritten.files_created)
        .map_err(|e| e.to_string())
}

/// A sorted delete-rewrite, file by file: with nothing to spill to, the big
/// files' sorts are refused after the small files were already rewritten, and
/// those rewritten files are removed: the catalog and the files are as they
/// were. With a disk manager the same rewrite spills and every file is
/// rewritten, each in `(k % 7)` order.
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_rewrite_refused_on_a_later_file_leaves_nothing() {
    let lake = compaction_lake().await;
    // A deletion in each file: `id` 1 and 8000 in the small ones, `k` 3 in the
    // big ones (all of whose `id`s are 0).
    for deleted in ["id = 1", "id = 8000", "k = 3 AND id = 0"] {
        lake.sql(&format!("DELETE FROM lake.main.t WHERE {deleted}"))
            .await
            .unwrap();
    }
    let before = (committed(&lake, "t"), on_disk(&lake, "t"));
    let refused = rewrite(&lake, runtime(SMALL_POOL, None))
        .await
        .expect_err("a big file's sort does not fit the pool");
    assert!(refused.contains("Resources exhausted"), "{refused}");
    assert_eq!(
        (committed(&lake, "t"), on_disk(&lake, "t")),
        before,
        "nothing left"
    );

    let factory = CountingFactory::new(None);
    assert_eq!(
        rewrite(&lake, runtime(SMALL_POOL, Some(Arc::clone(&factory)))).await,
        Ok(4)
    );
    assert!(factory.created() > 0, "the rewrite's sorts spilled");
    for file in layout(&lake, "t") {
        let key = |row: &(i64, Option<i64>, String)| row.1.map_or(i64::MAX, |k| k % 7);
        assert!(
            file.2.windows(2).all(|w| key(&w[0]) <= key(&w[1])),
            "in `(k % 7)` order"
        );
    }
}

/// What a tab inserts (ADR-0037's harness, `spikes/ducklake-large` in ferrion):
/// `id`, `k = id % 1000` and a 45-character string, 986,895 rows, 61 MiB in
/// Arrow, into a table sorted by `k` (the plan's sort) or by `(k + 1)` (the
/// writer's), under a 256 MiB pool with no disk manager: the sort holds the
/// rows in memory, and the write beside it is charged what it holds. On the
/// fork before ADR-0037 slice 4, both were refused at the pool, at 99 and
/// 179 MiB of a tab's memory.
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_insert_that_fits_the_pool_is_not_refused_by_its_charge() {
    for key in ["k", "(k + 1)"] {
        let lake = sqlite()(None).await;
        sorted_table(
            &lake,
            "t",
            &[(key, SortDirection::Asc, NullOrder::NullsLast)],
            true,
        )
        .await;
        let mut options = DuckLakeWriteOptions::default();
        options.data_inlining_row_limit = Some(10);
        let ctx = session(&lake, &options, runtime(256 * MIB, None), &input(1, 1));
        run(
            &ctx,
            "INSERT INTO lake.main.t SELECT CAST(value AS BIGINT) AS id, \
             CAST(value % 1000 AS BIGINT) AS k, \
             repeat(CAST((value * 2654435761) % 1000000007 AS VARCHAR), 5) AS v \
             FROM generate_series(1, 986895) AS t(value)",
        )
        .await
        .unwrap_or_else(|e| panic!("`{key}`: {e}"));
        let keys: Vec<Option<i64>> = layout(&lake, "t")
            .into_iter()
            .flat_map(|file| file.2.into_iter().map(|row| row.1))
            .collect();
        assert_eq!(keys.len(), 986_895, "`{key}`");
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "`{key}`: in key order"
        );
    }
}
