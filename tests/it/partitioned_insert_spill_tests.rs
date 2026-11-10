//! ferrion#754 (ADR-0037 slice 4, its follow-up): a partitioned `INSERT` past
//! its memory pool streams through a spilling sort keyed by partition, and is
//! refused whole where it cannot.
//!
//! A partitioned `INSERT` collects its input and splits it whole. That is still
//! what it does wherever the pool takes the input. Past the pool, where the
//! runtime can spill, its rows go through DataFusion's spilling sort keyed by
//! partition (in the order the input first shows them), then by the writer's
//! sort keys when it sorts, then by arrival; and into a write session that
//! keeps one partition's file open at a time. The reference every layout here
//! is held to is that collected write: the same `INSERT` under an unbounded
//! pool.

#![cfg(feature = "write-sqlite")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow::array::{Array, AsArray, Int64Array, RecordBatch, StringViewArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::DuckLakeWriteOptions;
use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};
use object_store::local::LocalFileSystem;

use crate::inlining_abstraction_tests::{Lake, create_with, sqlite, sqlite_made_by_duckdb};
use crate::sort_expression_tests::raw;
use crate::sorted_insert_spill_tests::{
    CountingFactory, FailingStore, committed, on_disk, run, runtime, spills,
};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;

/// The pool a spilling run is held to: a fraction of what it sorts. One open
/// file of these four columns holds about 4 MiB of `PARQUET_2_0` encoders, so
/// a session with every partition open (24 here) would hold about 100 MiB.
const SMALL_POOL: usize = 16 * MIB;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("p", DataType::Int64, true),
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Utf8View, true),
    ]))
}

/// `rows` rows in batches of `per_batch`: `id` the arrival position, `p` the
/// partition key (23 values scrambled across every batch, NULL every 17th
/// row), `k` a scrambled sort key with many ties and a NULL every 13th row,
/// `v` 40 characters that do not repeat. [`DUCKDB_INPUT`] is the same rows.
fn input(rows: i64, per_batch: i64) -> Vec<RecordBatch> {
    (0..rows)
        .step_by(per_batch as usize)
        .map(|start| {
            let ids = start..(start + per_batch).min(rows);
            let p: Int64Array = ids
                .clone()
                .map(|i| (i % 17 != 0).then_some((i * 31) % 23))
                .collect();
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
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(p),
                    Arc::new(k),
                    Arc::new(v),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// [`input`]'s rows, in DuckDB's SQL.
const DUCKDB_INPUT: &str = "SELECT id, \
     CASE WHEN id % 17 = 0 THEN NULL ELSE (id * 31) % 23 END AS p, \
     CASE WHEN id % 13 = 0 THEN NULL ELSE (id * 7919) % 1009 END AS k, \
     lpad(CAST((id * 2654435761) % 1000000007 AS VARCHAR), 40, '0') AS v \
     FROM range(0, 120000) AS t(id) ORDER BY id";

/// Row groups of 1,000 rows and a 16 KiB target, so each partition rolls into
/// several files; `limit` rows inline.
fn options(limit: usize) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.max_row_group_rows = Some(1000);
    options.target_file_size = Some(16 * KIB);
    options.data_inlining_row_limit = Some(limit);
    options
}

/// How the table orders its rows.
#[derive(Debug, Clone, Copy)]
enum Order {
    /// No sort order.
    None,
    /// `SET SORTED BY` this expression (as DuckDB stores it), with
    /// `sort_on_insert` as given.
    Sorted(&'static str, bool),
}

/// `main.<name>(id, p, k, v)`, partitioned by `p` and ordered by `order`.
async fn partitioned_table(lake: &Lake, name: &str, order: Order) -> i64 {
    create_with(
        lake,
        name,
        &[
            ("id", DataType::Int64),
            ("p", DataType::Int64),
            ("k", DataType::Int64),
            ("v", DataType::Utf8),
        ],
    )
    .unwrap();
    let table_id = lake.table_id(name);
    lake.writer
        .set_partition_spec(table_id, &[("p".to_string(), PartitionTransform::Identity)])
        .unwrap();
    if let Order::Sorted(expression, sort_on_insert) = order {
        let field = SortField {
            sort_key_index: 0,
            expression: "id".to_string(),
            dialect: "duckdb".to_string(),
            direction: SortDirection::Asc,
            null_order: NullOrder::NullsLast,
        };
        lake.writer.set_sort_spec(table_id, &[field]).unwrap();
        raw(
            lake,
            &format!(
                "UPDATE ducklake_sort_expression SET expression = '{}' WHERE table_id = {table_id}",
                expression.replace('\'', "''")
            ),
        )
        .await;
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
    }
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

/// One row as a file holds it: `(id, k)`.
type Row = (i64, Option<i64>);

/// `main.<table>`'s data files in data file id order, each read from disk:
/// `(its partition value, its rows in physical order)`.
fn files(lake: &Lake, table: &str) -> Vec<(Option<String>, Vec<Row>)> {
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
            let mut values = Vec::new();
            for batch in reader {
                let batch = batch.unwrap();
                let ids = batch.column(0).as_primitive::<Int64Type>();
                let ps = batch.column(1).as_primitive::<Int64Type>();
                let ks = batch.column(2).as_primitive::<Int64Type>();
                for row in 0..batch.num_rows() {
                    values.push(ps.is_valid(row).then(|| ps.value(row).to_string()));
                    rows.push((ids.value(row), ks.is_valid(row).then(|| ks.value(row))));
                }
            }
            values.dedup();
            assert!(values.len() <= 1, "a file holds one partition");
            (values.pop().flatten(), rows)
        })
        .collect()
}

/// [`files`], each run of files of one partition joined: `(the partition,
/// its rows in file and physical order)`, the partitions in file order.
fn runs(files: Vec<(Option<String>, Vec<Row>)>) -> Vec<(Option<String>, Vec<Row>)> {
    let mut out: Vec<(Option<String>, Vec<Row>)> = Vec::new();
    for (value, rows) in files {
        match out.last_mut() {
            Some((last, joined)) if *last == value => joined.extend(rows),
            _ => out.push((value, rows)),
        }
    }
    out
}

/// A DuckDB connection with `lake` attached as `lake`, on one thread.
fn duckdb_on(lake: &Lake) -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite; SET threads = 1;
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/');",
            lake.data.parent().unwrap().join("lake.db").display(),
            lake.data.display()
        ))
        .unwrap();
    connection
}

/// `main.<table>` as DuckDB reads it: each partition's rows in row id order,
/// each as `(key, id)` (`key` the value of the expression `key`), by
/// partition.
fn duckdb_partitions(
    connection: &duckdb::Connection,
    table: &str,
    key: &str,
) -> BTreeMap<Option<i64>, Vec<(Option<i64>, i64)>> {
    let mut statement = connection
        .prepare(&format!(
            "SELECT p, CAST({key} AS BIGINT), id FROM lake.main.{table} ORDER BY p NULLS LAST, rowid"
        ))
        .unwrap();
    let mut result = statement.query([]).unwrap();
    let mut out: BTreeMap<Option<i64>, Vec<(Option<i64>, i64)>> = BTreeMap::new();
    while let Some(row) = result.next().unwrap() {
        out.entry(row.get(0).unwrap())
            .or_default()
            .push((row.get(1).unwrap(), row.get(2).unwrap()));
    }
    out
}

/// Each partition's keys in their order, and its ids in id order: what two
/// layouts sorted by the same key share whatever their ties.
type Keyed = BTreeMap<Option<i64>, (Vec<Option<i64>>, Vec<i64>)>;

fn keys_and_ids(partitions: BTreeMap<Option<i64>, Vec<(Option<i64>, i64)>>) -> Keyed {
    partitions
        .into_iter()
        .map(|(p, rows)| {
            let keys = rows.iter().map(|r| r.0).collect();
            let mut ids: Vec<i64> = rows.iter().map(|r| r.1).collect();
            ids.sort_unstable();
            (p, (keys, ids))
        })
        .collect()
}

/// [`runs`]' rows, by partition, as [`duckdb_partitions`] keys them (`key`
/// `k` itself).
fn rows_by_partition(
    runs: &[(Option<String>, Vec<Row>)],
) -> BTreeMap<Option<i64>, Vec<(Option<i64>, i64)>> {
    runs.iter()
        .map(|(value, rows)| {
            (
                value.as_ref().map(|v| v.parse().unwrap()),
                rows.iter().map(|row| (row.1, row.0)).collect(),
            )
        })
        .collect()
}

/// A partitioned INSERT past a small pool, with a disk manager, spills (the
/// insert's own sort reports it) and lays the rows out as the collected write
/// does under an unbounded pool: each partition's files together, the
/// partitions in the same order, each partition's rows in the same order
/// (sorted as the table sorts them, ties in arrival order). Only where each
/// partition's files roll may differ: the collected write cuts a partition's
/// rows at the input's batches, the streamed one at the sort's. DuckDB reads
/// it back with each partition's rows in the same order, and, writing the same
/// rows itself on one thread, lays each partition out the same way.
///
/// Each order a partitioned table can have: none; an expression key with
/// `sort_on_insert` on (the writer's sort); a bare column with it on (the
/// plan's sort, whose order the partition sort keeps); a bare column with it
/// off while inlining is enabled (the writer's sort again).
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_insert_past_the_pool_spills_and_lays_out_as_the_collected_write() {
    let batches = input(120_000, 997);
    let cases: [(Order, usize); 4] = [
        (Order::None, 0),
        (Order::Sorted("(k % 7)", true), 0),
        (Order::Sorted("k", true), 0),
        (Order::Sorted("k", false), 10),
    ];
    for (order, limit) in cases {
        let lake = sqlite_made_by_duckdb()(None).await;
        let options = options(limit);
        for table in ["b", "c"] {
            partitioned_table(&lake, table, order).await;
        }
        // The reference: the collected write, under an unbounded pool.
        let ctx = session(&lake, &options, RuntimeEnvBuilder::new(), &batches);
        let plan = run(&ctx, "INSERT INTO lake.main.c SELECT id, p, k, v FROM src")
            .await
            .unwrap_or_else(|e| panic!("{order:?}: {e}"));
        assert_eq!(spills(&plan).0, 0, "{order:?}: in memory");
        let collected = files(&lake, "c");
        assert!(
            collected.len() > 2 * 24,
            "{order:?}: each partition rolls into files ({})",
            collected.len()
        );
        let expected = runs(collected);
        assert_eq!(
            expected.len(),
            24,
            "{order:?}: one run of files a partition"
        );

        let factory = CountingFactory::new(None);
        let ctx = session(
            &lake,
            &options,
            runtime(SMALL_POOL, Some(Arc::clone(&factory))),
            &batches,
        );
        let plan = run(&ctx, "INSERT INTO lake.main.b SELECT id, p, k, v FROM src")
            .await
            .unwrap_or_else(|e| panic!("{order:?}: {e}"));
        assert!(spills(&plan).0 > 0, "{order:?}: the insert's sort spilled");
        assert!(factory.created() > 0, "{order:?}: spill files were made");
        // Rolled where the sort's batches end (a partition's rows can come as
        // one batch of the plan's sort, of 8,192 rows), so maybe less often.
        let written = files(&lake, "b");
        assert!(
            written.len() > 24,
            "{order:?}: partitions roll into files ({})",
            written.len()
        );
        let got = runs(written);
        assert_eq!(got.len(), 24, "{order:?}: each partition's files together");
        match order {
            // The plan's sort is DataFusion's, which is not stable, in memory
            // or spilled: ties on `k` may fall otherwise, and so may which
            // partition's row comes first (the partitions are in the order
            // the plan's sorted rows first show them). Each partition's keys,
            // and its rows, are the same.
            Order::Sorted("k", true) => {
                let by_value = |runs: &[(Option<String>, Vec<Row>)]| {
                    let mut runs = runs.to_vec();
                    runs.sort_by(|a, b| a.0.cmp(&b.0));
                    runs
                };
                let (got, expected) = (by_value(&got), by_value(&expected));
                for ((value, rows), (want_value, want_rows)) in got.iter().zip(&expected) {
                    assert_eq!(value, want_value, "{order:?}: the same partitions");
                    let keys = |rows: &[Row]| rows.iter().map(|r| r.1).collect::<Vec<_>>();
                    assert_eq!(
                        keys(rows),
                        keys(want_rows),
                        "{order:?}: {value:?} in `k` order"
                    );
                    let mut ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
                    let mut want: Vec<i64> = want_rows.iter().map(|r| r.0).collect();
                    ids.sort_unstable();
                    want.sort_unstable();
                    assert_eq!(ids, want, "{order:?}: {value:?}'s rows");
                }
            },
            _ => assert!(
                got == expected,
                "{order:?}: laid out as the collected write"
            ),
        }

        // DuckDB reads each partition's rows in the order they were written.
        let duckdb = duckdb_on(&lake);
        assert_eq!(
            duckdb_partitions(&duckdb, "b", "k"),
            rows_by_partition(&got),
            "{order:?}: DuckDB reads the streamed write"
        );
        // And DuckDB, writing the same rows on one thread into a table of the
        // same partitioning and order, puts each partition's rows in the same
        // order: exactly, unsorted; by the sort key, sorted, its ties being its
        // sort's (DuckDB's sort keeps no arrival order, even on one thread).
        let (sort, key) = match order {
            Order::Sorted(key, sort_on_insert) => (
                format!(
                    "ALTER TABLE lake.main.o SET SORTED BY ({key});
                     CALL lake.set_option('sort_on_insert', '{sort_on_insert}', table_name => 'o');"
                ),
                key,
            ),
            Order::None => (String::new(), "k"),
        };
        duckdb
            .execute_batch(&format!(
                "CALL lake.set_option('data_inlining_row_limit', {limit});
                 CREATE TABLE lake.main.o (id BIGINT, p BIGINT, k BIGINT, v VARCHAR);
                 ALTER TABLE lake.main.o SET PARTITIONED BY (p);
                 {sort}
                 INSERT INTO lake.main.o {DUCKDB_INPUT};"
            ))
            .unwrap_or_else(|e| panic!("{order:?}: {e}"));
        let (ours, duckdbs) = (
            duckdb_partitions(&duckdb, "b", key),
            duckdb_partitions(&duckdb, "o", key),
        );
        match order {
            Order::None => assert_eq!(
                duckdbs, ours,
                "{order:?}: each partition as DuckDB lays it out"
            ),
            Order::Sorted(..) => assert_eq!(
                keys_and_ids(duckdbs),
                keys_and_ids(ours),
                "{order:?}: each partition in DuckDB's key order"
            ),
        }

        // An OVERWRITE past the pool replaces the rows, as the collected one does.
        if matches!(order, Order::None) {
            let ctx = session(
                &lake,
                &options,
                runtime(SMALL_POOL, Some(CountingFactory::new(None))),
                &batches,
            );
            run(
                &ctx,
                "INSERT OVERWRITE lake.main.b SELECT id, p, k, v FROM src",
            )
            .await
            .unwrap();
            assert!(
                runs(files(&lake, "b")) == expected,
                "an overwrite leaves the rows once"
            );
        }
    }
}

/// A partitioned INSERT that fails past the pool leaves nothing: no catalog
/// row and no file. Refused at the pool with no disk manager to spill to; past
/// its spill budget; by its input, failing on its last rows; and by its
/// write, the fourth file's upload failing after the sort.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_insert_refused_past_the_pool_leaves_nothing() {
    let batches = input(120_000, 997);
    let insert = "INSERT INTO lake.main.b SELECT id, p, k, v FROM src";
    let failing = "INSERT INTO lake.main.b SELECT id, p, \
                   CASE WHEN id = 119990 THEN CAST(concat('x', v) AS BIGINT) ELSE k END, v FROM src";
    let failing_store = || -> Arc<dyn object_store::ObjectStore> {
        Arc::new(FailingStore {
            inner: LocalFileSystem::new(),
            allowed: 3,
            started: AtomicUsize::new(0),
        })
    };
    type Case<'a> = (
        &'a str,
        Order,
        RuntimeEnvBuilder,
        Option<Arc<dyn object_store::ObjectStore>>,
        &'a str,
        &'a str,
    );
    let cases: Vec<Case<'_>> = vec![
        (
            "the pool, no disk manager",
            Order::None,
            runtime(SMALL_POOL, None),
            None,
            insert,
            "Resources exhausted",
        ),
        (
            "the spill budget",
            Order::Sorted("(k % 7)", true),
            runtime(
                SMALL_POOL,
                Some(CountingFactory::new(Some(64 * KIB as u64))),
            ),
            None,
            insert,
            "exceeded",
        ),
        (
            "the input",
            Order::None,
            runtime(SMALL_POOL, Some(CountingFactory::new(None))),
            None,
            failing,
            "Cannot cast",
        ),
        (
            "the write",
            Order::Sorted("(k % 7)", true),
            runtime(SMALL_POOL, Some(CountingFactory::new(None))),
            Some(failing_store()),
            insert,
            "injected upload failure",
        ),
    ];
    for (what, order, runtime, store, sql, says) in cases {
        let lake = sqlite()(None).await;
        partitioned_table(&lake, "b", order).await;
        let before = committed(&lake, "b");
        let ctx = session(&lake, &options(0), runtime, &batches);
        if let Some(store) = store {
            ctx.runtime_env().register_object_store(
                datafusion::execution::object_store::ObjectStoreUrl::local_filesystem().as_ref(),
                store,
            );
        }
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

/// A partitioned INSERT of no more rows than the table inlines is decided
/// whole, as before: past the pool it is refused, not streamed, even with a
/// disk manager (the collected path would have inlined the rows in their
/// arrival order, which the partition sort does not keep).
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_insert_it_could_inline_is_refused_past_the_pool() {
    // Eight rows of 1 MiB strings: past a 4 MiB pool, within a limit of 10.
    let batches: Vec<RecordBatch> = (0..8i64)
        .map(|i| {
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(vec![i])),
                    Arc::new(Int64Array::from(vec![(i * 5) % 3])),
                    Arc::new(Int64Array::from(vec![i])),
                    Arc::new(StringViewArray::from(vec!["x".repeat(MIB)])),
                ],
            )
            .unwrap()
        })
        .collect();
    let lake = sqlite()(None).await;
    partitioned_table(&lake, "b", Order::None).await;
    let before = committed(&lake, "b");
    let factory = CountingFactory::new(None);
    let ctx = session(
        &lake,
        &options(10),
        runtime(4 * MIB, Some(Arc::clone(&factory))),
        &batches,
    );
    let error = run(&ctx, "INSERT INTO lake.main.b SELECT id, p, k, v FROM src")
        .await
        .expect_err("refused at the pool");
    assert!(error.contains("Resources exhausted"), "{error}");
    // Refused where the collected path holds the input: no sort began.
    assert!(!error.contains("ExternalSorter"), "{error}");
    assert_eq!(factory.created(), 0, "nothing spilled");
    assert_eq!(committed(&lake, "b"), before, "nothing committed");
    assert_eq!(on_disk(&lake, "b"), Vec::<PathBuf>::new(), "no file left");
}

/// A partitioned INSERT past the pool whose table's partition spec changed
/// between planning and execution is refused as a conflict, and leaves nothing:
/// its rows were sorted by the spec it was planned under.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_insert_past_the_pool_refuses_a_spec_changed_since_planning() {
    let batches = input(120_000, 997);
    let lake = sqlite()(None).await;
    let table_id = partitioned_table(&lake, "b", Order::None).await;
    let ctx = session(
        &lake,
        &options(0),
        runtime(SMALL_POOL, Some(CountingFactory::new(None))),
        &batches,
    );
    let plan = ctx
        .sql("INSERT INTO lake.main.b SELECT id, p, k, v FROM src")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    lake.writer
        .set_partition_spec(table_id, &[("k".to_string(), PartitionTransform::Identity)])
        .unwrap();
    let before = committed(&lake, "b");
    let error = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect_err("a conflict")
        .to_string();
    assert!(error.contains("retry the write"), "{error}");
    assert_eq!(committed(&lake, "b"), before, "nothing committed");
    assert_eq!(on_disk(&lake, "b"), Vec::<PathBuf>::new(), "no file left");
}

/// A sort key that fails over one partition's rows only (`k / (p - 5)`, a
/// division by zero where `p` is 5) leaves that partition's rows in arrival
/// order and every other partition sorted, past the pool as under it: the
/// collected write sorts each partition on its own.
#[tokio::test(flavor = "multi_thread")]
async fn a_key_failing_in_one_partition_leaves_only_that_one_unsorted() {
    let batches = input(120_000, 997);
    let order = Order::Sorted("(k / (p - 5))", true);
    let lake = sqlite()(None).await;
    for table in ["b", "c"] {
        partitioned_table(&lake, table, order).await;
    }
    let ctx = session(&lake, &options(0), RuntimeEnvBuilder::new(), &batches);
    run(&ctx, "INSERT INTO lake.main.c SELECT id, p, k, v FROM src")
        .await
        .unwrap();
    let expected = runs(files(&lake, "c"));
    let factory = CountingFactory::new(None);
    let ctx = session(
        &lake,
        &options(0),
        runtime(SMALL_POOL, Some(Arc::clone(&factory))),
        &batches,
    );
    let plan = run(&ctx, "INSERT INTO lake.main.b SELECT id, p, k, v FROM src")
        .await
        .unwrap();
    assert!(spills(&plan).0 > 0, "the insert's sort spilled");
    let got = runs(files(&lake, "b"));
    assert!(got == expected, "laid out as the collected write");
    // The reference itself: partition 5 in arrival order, another sorted.
    let ids = |value: &str| -> Vec<i64> {
        got.iter()
            .find(|(v, _)| v.as_deref() == Some(value))
            .unwrap()
            .1
            .iter()
            .map(|r| r.0)
            .collect()
    };
    let five = ids("5");
    assert!(
        five.windows(2).all(|w| w[0] < w[1]),
        "p = 5 in arrival order"
    );
    let six = ids("6");
    assert!(
        !six.windows(2).all(|w| w[0] < w[1]),
        "p = 6 sorted by its key, not in arrival order"
    );
}
