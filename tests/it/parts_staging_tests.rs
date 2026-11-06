//! `Staging::Parts` (ADR-0037 in ferrion, Decision 1): a data file goes out to its
//! multipart upload a part at a time while it is written, rather than held whole
//! until it is finished, and the statistics its catalog row carries, taken from
//! the footer the writer returned, equal those of the footer read back.
//!
//! Run natively over a store that logs every part, on the local filesystem so
//! DuckDB can read the lake back. The catalog is SQLite.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{MultipartUpload, ObjectStore, PutPayload, UploadPart};
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::partition::PartitionTransform;
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter, SqliteMetadataWriter, Staging,
    WriteMode,
};

/// Parts this small make a file of a few hundred KiB go out in many of them.
const PART_BYTES: usize = 16 * 1024;

/// What the store was asked to do, in order.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Open(String),
    Part(String, usize),
    Complete(String),
    Abort(String),
    Put(String, usize),
}

type Log = Arc<Mutex<Vec<Event>>>;

/// What the store does to one part.
#[derive(Debug, Clone, Copy)]
enum Fault {
    /// Refuses the Nth part sent, counting every upload's.
    Fail(usize),
    /// Never answers the `part`th part (from 1) of the `upload`th upload opened
    /// (from 1), so the write sending it can only be cancelled.
    Hang {
        upload: usize,
        part: usize,
    },
}

/// A store over the local filesystem that logs every write, and can fail or
/// hang the Nth part.
#[derive(Debug)]
struct PartsStore {
    inner: Arc<dyn ObjectStore>,
    log: Log,
    parts: Arc<AtomicUsize>,
    opened: AtomicUsize,
    fault: Option<Fault>,
}

impl PartsStore {
    fn new(fault: Option<Fault>) -> Self {
        Self {
            inner: Arc::new(LocalFileSystem::new()),
            log: Arc::default(),
            parts: Arc::default(),
            opened: AtomicUsize::new(0),
            fault,
        }
    }

    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }
}

impl std::fmt::Display for PartsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PartsStore({})", self.inner)
    }
}

#[derive(Debug)]
struct LoggedUpload {
    inner: Box<dyn MultipartUpload>,
    path: String,
    log: Log,
    parts: Arc<AtomicUsize>,
    /// This upload's place in open order, from 1, and its parts sent so far.
    index: usize,
    own_parts: usize,
    fault: Option<Fault>,
}

#[async_trait::async_trait]
impl MultipartUpload for LoggedUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let part = self.parts.fetch_add(1, Ordering::SeqCst) + 1;
        self.own_parts += 1;
        match self.fault {
            Some(Fault::Fail(n)) if n == part => {
                return Box::pin(async {
                    Err(object_store::Error::Generic {
                        store: "PartsStore",
                        source: "injected part failure".into(),
                    })
                });
            },
            Some(Fault::Hang {
                upload,
                part,
            }) if upload == self.index && part == self.own_parts => {
                return Box::pin(futures::future::pending());
            },
            _ => {},
        }
        self.log
            .lock()
            .unwrap()
            .push(Event::Part(self.path.clone(), data.content_length()));
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> object_store::Result<object_store::PutResult> {
        let result = self.inner.complete().await;
        self.log
            .lock()
            .unwrap()
            .push(Event::Complete(self.path.clone()));
        result
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(Event::Abort(self.path.clone()));
        self.inner.abort().await
    }
}

#[async_trait::async_trait]
impl ObjectStore for PartsStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.log
            .lock()
            .unwrap()
            .push(Event::Put(location.to_string(), payload.content_length()));
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.log
            .lock()
            .unwrap()
            .push(Event::Open(location.to_string()));
        Ok(Box::new(LoggedUpload {
            inner: self.inner.put_multipart_opts(location, opts).await?,
            path: location.to_string(),
            log: Arc::clone(&self.log),
            parts: Arc::clone(&self.parts),
            index: self.opened.fetch_add(1, Ordering::SeqCst) + 1,
            own_parts: 0,
            fault: self.fault,
        }))
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("v", DataType::Float64, true),
    ]))
}

/// `count` batches of 2,000 rows: ascending ids, names that vary in length (some
/// NULL), and values with NULLs and, in one batch only, a NaN, so the files'
/// statistics differ in bounds, null counts and NaN presence.
fn batches(count: i64) -> Vec<RecordBatch> {
    (0..count)
        .map(|b| {
            let ids: Vec<i64> = (b * 2000..(b + 1) * 2000).collect();
            let names: Vec<Option<String>> = ids
                .iter()
                .map(|id| {
                    (id % 11 != 0).then(|| {
                        format!(
                            "n{}-{}",
                            id * 7919 % 100_003,
                            "x".repeat((id % 13) as usize)
                        )
                    })
                })
                .collect();
            let values: Vec<Option<f64>> = ids
                .iter()
                .map(|id| match id {
                    id if id % 17 == 0 => None,
                    id if *id == 7 * 2000 + 5 => Some(f64::NAN),
                    id => Some((*id as f64) * 0.5 - 1000.0),
                })
                .collect();
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(StringArray::from(names)),
                    Arc::new(Float64Array::from(values)),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// Row groups of 1,000 rows and a 256 KiB target, so 30 batches roll into
/// several files of many parts each; nothing inlined.
fn options(staging: Staging) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.max_row_group_rows = Some(1000);
    options.target_file_size = Some(256 * 1024);
    options.data_inlining_row_limit = Some(0);
    options.staging = Some(staging);
    options
}

struct Lake {
    temp: TempDir,
    writer: Arc<SqliteMetadataWriter>,
    store: Arc<PartsStore>,
}

impl Lake {
    async fn new(fault: Option<Fault>) -> Self {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        // DuckDB makes the catalog, so it is the schema DuckDB reads back.
        crate::common::ensure_ducklake_installed();
        crate::common::ensure_extension_installed("sqlite");
        duckdb::Connection::open_in_memory()
            .unwrap()
            .execute_batch(&format!(
                "LOAD ducklake; LOAD sqlite; \
                 ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0); \
                 DETACH lake;",
                temp.path().join("test.db").display(),
                data.display()
            ))
            .unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&self::url(&temp))
            .await
            .unwrap()
            .with_data_inlining_row_limit(0);
        Self {
            temp,
            writer: Arc::new(writer),
            store: Arc::new(PartsStore::new(fault)),
        }
    }

    fn table_writer(&self, staging: Staging) -> DuckLakeTableWriter {
        let store: Arc<dyn ObjectStore> = self.store.clone();
        DuckLakeTableWriter::new(self.writer.clone(), store)
            .unwrap()
            .with_options(&options(staging))
    }

    fn catalog(&self) -> std::path::PathBuf {
        self.temp.path().join("test.db")
    }

    fn data(&self) -> std::path::PathBuf {
        self.temp.path().join("data")
    }

    /// Every live data file's catalog row and its column statistics, in file
    /// order: what a reader prunes by.
    async fn catalog_stats(&self) -> Vec<Vec<Option<String>>> {
        let pool = SqlitePool::connect(&url(&self.temp)).await.unwrap();
        let columns = [
            "f.record_count",
            "f.file_size_bytes",
            "f.footer_size",
            "s.column_id",
            "s.column_size_bytes",
            "s.value_count",
            "s.null_count",
            "s.min_value",
            "s.max_value",
            "s.contains_nan",
            "s.extra_stats",
        ];
        let select: Vec<String> = columns
            .iter()
            .map(|c| format!("CAST({c} AS TEXT)"))
            .collect();
        let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
            "SELECT {} \
             FROM ducklake_data_file f \
             JOIN ducklake_file_column_stats s USING (data_file_id) \
             WHERE f.end_snapshot IS NULL \
             ORDER BY f.data_file_id, s.column_id",
            select.join(", ")
        )))
        .fetch_all(&pool)
        .await
        .unwrap();
        pool.close().await;
        rows.iter()
            .map(|row| {
                (0..columns.len())
                    .map(|i| row.get::<Option<String>, _>(i))
                    .collect()
            })
            .collect()
    }

    /// The live data files' paths, with the size the catalog records.
    async fn files(&self) -> Vec<(String, i64)> {
        let pool = SqlitePool::connect(&url(&self.temp)).await.unwrap();
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT path, file_size_bytes FROM ducklake_data_file \
             WHERE end_snapshot IS NULL ORDER BY data_file_id",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        pool.close().await;
        rows
    }

    /// Every Parquet file under the data directory.
    fn parquet_on_disk(&self) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();
        let mut pending = vec![self.data()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    found.push(path);
                }
            }
        }
        found
    }

    /// DuckDB's own read of the table: row count, sums and counts that touch
    /// every column, and a range read that prunes by the files' statistics.
    fn duckdb_reads(&self) -> (i64, i64, i64, i64, i64, i64) {
        let oracle = duckdb::Connection::open_in_memory().unwrap();
        oracle
            .execute_batch(&format!(
                "LOAD ducklake; LOAD sqlite; \
                 ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', READ_ONLY);",
                self.catalog().display(),
                self.data().display()
            ))
            .unwrap();
        oracle
            .query_row(
                "SELECT count(*), sum(id)::BIGINT, count(name), sum(length(name))::BIGINT, \
                        count(v), (SELECT count(*) FROM lake.main.t WHERE id BETWEEN 20000 AND 20999) \
                 FROM lake.main.t",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
            )
            .unwrap()
    }
}

fn url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

/// What [`Lake::duckdb_reads`] must return for `batches(count)`.
fn expected_reads(count: i64) -> (i64, i64, i64, i64, i64, i64) {
    let mut out = (0, 0, 0, 0, 0, 1000);
    for batch in batches(count) {
        out.0 += batch.num_rows() as i64;
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        out.1 += ids.iter().flatten().sum::<i64>();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        out.2 += names.iter().flatten().count() as i64;
        out.3 += names
            .iter()
            .flatten()
            .map(|n| n.chars().count() as i64)
            .sum::<i64>();
        let values = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        out.4 += values.iter().flatten().count() as i64;
    }
    out
}

fn parts_of(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Event::Part(..)))
        .count()
}

/// A streaming session under `Staging::Parts` sends each file's parts while it is
/// written: before `finish`, parts have gone out for every file rolled so far and
/// for the one still open, whose upload is not complete. Every object is exactly
/// its parts. The statistics the catalog records equal those of the same rows
/// written under `Staging::Memory`, whose statistics come from the footer read
/// back from its bytes. DuckDB reads the lake.
#[tokio::test(flavor = "multi_thread")]
async fn parts_go_out_while_a_session_writes_and_the_footer_stats_agree() {
    let lake = Lake::new(None).await;
    let mut session = lake
        .table_writer(Staging::Parts {
            part_bytes: PART_BYTES,
        })
        .begin_write("main", "t", &schema(), WriteMode::Create)
        .unwrap();
    for batch in batches(30) {
        session.write_batch_async(&batch).await.unwrap();
    }

    let before = lake.store.events();
    let opened: Vec<&String> = before
        .iter()
        .filter_map(|e| match e {
            Event::Open(path) => Some(path),
            _ => None,
        })
        .collect();
    let completed = |path: &String| before.contains(&Event::Complete(path.clone()));
    assert!(
        opened.len() >= 3,
        "a file per roll has its upload open before finish: {opened:?}"
    );
    assert!(
        parts_of(&before) >= 2 * opened.len(),
        "{} parts before finish for {} files",
        parts_of(&before),
        opened.len()
    );
    let open_file = opened.last().unwrap();
    assert!(
        !completed(open_file),
        "the file still being written is not complete"
    );
    assert!(
        before
            .iter()
            .any(|e| matches!(e, Event::Part(path, _) if path == *open_file)),
        "the file still being written has sent parts already"
    );
    assert!(
        !before.iter().any(|e| matches!(e, Event::Put(..))),
        "no file went out whole: {before:?}"
    );

    session.finish().await.unwrap();

    let events = lake.store.events();
    let files = lake.files().await;
    assert_eq!(files.len(), opened.len(), "one file per upload");
    for (path, size) in &files {
        let key = events
            .iter()
            .find_map(|e| match e {
                Event::Open(key) if key.ends_with(path.as_str()) => Some(key.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{path} went out in parts"));
        let sent: usize = events
            .iter()
            .filter_map(|e| match e {
                Event::Part(p, len) if *p == key => Some(*len),
                _ => None,
            })
            .sum();
        assert_eq!(sent as i64, *size, "{path}: the object is its parts");
        assert!(
            events.contains(&Event::Complete(key.clone())),
            "{path} completed"
        );
        let on_disk = std::fs::metadata(format!("/{key}")).unwrap().len();
        assert_eq!(on_disk as i64, *size, "{path}: recorded size");
    }
    assert!(!events.iter().any(|e| matches!(e, Event::Abort(..))));

    // The same rows under memory staging, whose statistics are read back from
    // each finished file's footer.
    let memory = Lake::new(None).await;
    let mut session = memory
        .table_writer(Staging::Memory)
        .begin_write("main", "t", &schema(), WriteMode::Create)
        .unwrap();
    for batch in batches(30) {
        session.write_batch_async(&batch).await.unwrap();
    }
    session.finish().await.unwrap();

    let parts_stats = lake.catalog_stats().await;
    assert_eq!(
        parts_stats.len(),
        files.len() * 3,
        "every column of every file"
    );
    assert!(
        parts_stats.iter().any(|row| row[9].as_deref() == Some("1")),
        "one file records its NaN: {parts_stats:?}"
    );
    assert_eq!(
        parts_stats,
        memory.catalog_stats().await,
        "the footer the writer returned and the footer read back"
    );

    assert_eq!(lake.duckdb_reads(), expected_reads(30));
}

/// A tab's `CREATE TABLE … AS` path ([`DuckLakeTableWriter::create_table`]): each
/// file goes out in parts as it rolls, so the first file is complete in the store
/// before the last one is opened, and DuckDB reads the lake.
#[tokio::test(flavor = "multi_thread")]
async fn create_table_sends_each_file_in_parts_as_it_rolls() {
    let lake = Lake::new(None).await;
    lake.table_writer(Staging::Parts {
        part_bytes: PART_BYTES,
    })
    .create_table("main", "t", &batches(30))
    .await
    .unwrap();

    let events = lake.store.events();
    let opens: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Event::Open(_)).then_some(i))
        .collect();
    let files = lake.files().await;
    assert!(files.len() >= 3, "{files:?}");
    assert_eq!(opens.len(), files.len());
    let Event::Open(first) = &events[opens[0]] else {
        unreachable!()
    };
    let first_complete = events
        .iter()
        .position(|e| *e == Event::Complete(first.clone()))
        .unwrap();
    assert!(
        first_complete < *opens.last().unwrap(),
        "the first file is complete before the last is opened: {events:?}"
    );
    // That each file's parts precede its footer is `table_writer`'s unit test
    // `create_table_sends_parts_before_each_footer`.
    // Every file but the last sent parts before its footer: a file's parts all
    // come before its completion, and a file holds several.
    for open in &opens {
        let Event::Open(path) = &events[*open] else {
            unreachable!()
        };
        let parts = events
            .iter()
            .filter(|e| matches!(e, Event::Part(p, _) if p == path))
            .count();
        assert!(parts >= 2, "{path}: {parts} parts");
    }
    assert_eq!(lake.duckdb_reads(), expected_reads(30));
}

/// A part the store refuses fails the write and aborts that file's upload; an
/// aborted session then leaves no object and no catalog row.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_part_aborts_the_upload_and_leaves_nothing() {
    let lake = Lake::new(Some(Fault::Fail(5))).await;
    let mut session = lake
        .table_writer(Staging::Parts {
            part_bytes: PART_BYTES,
        })
        .begin_write("main", "t", &schema(), WriteMode::Create)
        .unwrap();
    let mut failure = None;
    for batch in batches(30) {
        if let Err(error) = session.write_batch_async(&batch).await {
            failure = Some(error);
            break;
        }
    }
    let failure = failure.expect("the fifth part is refused");
    assert!(
        failure.to_string().contains("injected part failure"),
        "{failure}"
    );
    session.abort().await.unwrap();

    let events = lake.store.events();
    let aborted = events
        .iter()
        .filter(|e| matches!(e, Event::Abort(_)))
        .count();
    let opened = events
        .iter()
        .filter(|e| matches!(e, Event::Open(_)))
        .count();
    let completed = events
        .iter()
        .filter(|e| matches!(e, Event::Complete(_)))
        .count();
    assert!(aborted >= 1, "{events:?}");
    assert_eq!(
        aborted + completed,
        opened,
        "every upload is completed or aborted: {events:?}"
    );
    assert_eq!(lake.parquet_on_disk(), Vec::<std::path::PathBuf>::new());
    assert!(lake.files().await.is_empty());
}

/// How many uploads were opened, completed and aborted.
fn counts(events: &[Event]) -> (usize, usize, usize) {
    let count = |f: fn(&Event) -> bool| events.iter().filter(|e| f(e)).count();
    (
        count(|e| matches!(e, Event::Open(_))),
        count(|e| matches!(e, Event::Complete(_))),
        count(|e| matches!(e, Event::Abort(_))),
    )
}

/// A write whose part is cancelled mid-send (its future dropped, here by a
/// timeout over a part the store never answers) has lost that part's bytes: its
/// `finish` is refused rather than writing a file with a hole in it, and the
/// refusal aborts the open file's upload and removes the files already
/// uploaded, before it returns. Run on one thread, so nothing a drop might
/// spawn runs before the checks.
#[tokio::test(flavor = "current_thread")]
async fn a_cancelled_part_refuses_finish_and_aborts_every_upload() {
    let lake = Lake::new(Some(Fault::Hang {
        upload: 3,
        part: 3,
    }))
    .await;
    let mut session = lake
        .table_writer(Staging::Parts {
            part_bytes: PART_BYTES,
        })
        .begin_write("main", "t", &schema(), WriteMode::Create)
        .unwrap();
    let mut cancelled = false;
    for batch in batches(30) {
        let write = session.write_batch_async(&batch);
        match tokio::time::timeout(std::time::Duration::from_millis(500), write).await {
            Ok(written) => written.unwrap(),
            Err(_) => {
                cancelled = true;
                break;
            },
        }
    }
    assert!(cancelled, "the third file's third part hangs");
    let before = lake.store.events();
    assert_eq!(counts(&before).0, 3, "{before:?}");

    let error = session.finish().await.unwrap_err();
    assert!(error.to_string().contains("cancelled"), "{error}");

    let events = lake.store.events();
    let (opened, completed, aborted) = counts(&events);
    assert!(
        aborted >= 1,
        "the open file's upload is aborted: {events:?}"
    );
    assert_eq!(
        opened,
        completed + aborted,
        "every upload completed or aborted: {events:?}"
    );
    assert_eq!(
        lake.parquet_on_disk(),
        Vec::<std::path::PathBuf>::new(),
        "no file and no half upload is left"
    );
    assert!(lake.files().await.is_empty());
}

/// With no tokio runtime on the thread, as a browser drives a write, a session
/// under `Staging::Parts` still sends its parts while it writes and uploads its
/// rolled files itself (`StagedUploads::upload_inline`): before `finish`, every
/// file but the open one is complete and the open one has sent parts. DuckDB
/// reads the lake.
#[tokio::test(flavor = "multi_thread")]
async fn parts_go_out_with_no_tokio_runtime() {
    let lake = Lake::new(None).await;
    let table_writer = lake.table_writer(Staging::Parts {
        part_bytes: PART_BYTES,
    });
    let store = Arc::clone(&lake.store);
    let (files, before) = std::thread::spawn(move || {
        assert!(tokio::runtime::Handle::try_current().is_err());
        futures::executor::block_on(async {
            let mut session = table_writer
                .begin_write("main", "t", &schema(), WriteMode::Create)
                .unwrap();
            for batch in batches(30) {
                session.write_batch_async(&batch).await.unwrap();
            }
            let before = store.events();
            (session.finish().await.unwrap().files_written, before)
        })
    })
    .join()
    .unwrap();

    let (opened, completed, aborted) = counts(&before);
    assert!(
        files >= 3 && opened == files,
        "{files} files, {opened} opened"
    );
    assert_eq!(
        completed,
        opened - 1,
        "all but the open file uploaded: {before:?}"
    );
    assert_eq!(aborted, 0);
    let Some(Event::Open(open_file)) = before.iter().rev().find(|e| matches!(e, Event::Open(_)))
    else {
        unreachable!()
    };
    assert!(
        before
            .iter()
            .any(|e| matches!(e, Event::Part(path, _) if path == open_file)),
        "the open file has sent parts"
    );
    assert_eq!(lake.duckdb_reads(), expected_reads(30));
}

fn partitioned_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("p", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]))
}

/// `count` batches of 2,000 rows, each holding all three partitions `p`.
fn partitioned_batches(count: i64) -> Vec<RecordBatch> {
    (0..count)
        .map(|b| {
            let ids: Vec<i64> = (b * 2000..(b + 1) * 2000).collect();
            RecordBatch::try_new(
                partitioned_schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids.iter().map(|id| id % 3))),
                    Arc::new(Int64Array::from(ids.clone())),
                    Arc::new(StringArray::from_iter_values(
                        ids.iter().map(|id| format!("s{}", id * 7919 % 100_003)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// A partitioned session under `Staging::Parts` sends parts for every partition
/// file it holds open: before `finish`, three files (one per partition) have
/// sent parts and are not complete. DuckDB reads every partition back.
#[tokio::test(flavor = "multi_thread")]
async fn a_partitioned_session_sends_every_open_files_parts() {
    let lake = Lake::new(None).await;
    let created = lake
        .table_writer(Staging::Memory)
        .create_table("main", "pt", &partitioned_batches(1)[..1])
        .await
        .unwrap();
    lake.writer
        .set_partition_spec(
            created.table_id,
            &[("p".to_string(), PartitionTransform::Identity)],
        )
        .unwrap();
    let mut options = options(Staging::Parts {
        part_bytes: PART_BYTES,
    });
    // One file per partition for the whole write: open together throughout.
    options.target_file_size = Some(64 * 1024 * 1024);
    let store: Arc<dyn ObjectStore> = lake.store.clone();
    let mut session = DuckLakeTableWriter::new(lake.writer.clone(), store)
        .unwrap()
        .with_options(&options)
        .begin_write("main", "pt", &partitioned_schema(), WriteMode::Append)
        .unwrap();
    for batch in partitioned_batches(30) {
        session.write_batch_async(&batch).await.unwrap();
    }
    let before = lake.store.events();
    let mut streaming: Vec<&String> = before
        .iter()
        .filter_map(|e| match e {
            Event::Part(path, _) if !before.contains(&Event::Complete(path.clone())) => Some(path),
            _ => None,
        })
        .collect();
    streaming.sort();
    streaming.dedup();
    assert_eq!(
        streaming.len(),
        3,
        "each partition's open file has sent parts: {streaming:?}"
    );
    session.finish().await.unwrap();

    let oracle = duckdb::Connection::open_in_memory().unwrap();
    oracle
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite; \
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', READ_ONLY);",
            lake.catalog().display(),
            lake.data().display()
        ))
        .unwrap();
    let mut statement = oracle
        .prepare("SELECT p, count(*), sum(id)::BIGINT FROM lake.main.pt GROUP BY p ORDER BY p")
        .unwrap();
    let rows: Vec<(i64, i64, i64)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<std::result::Result<_, _>>()
        .unwrap();
    let mut expected = vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)];
    for batch in partitioned_batches(30)
        .into_iter()
        .chain(partitioned_batches(1).into_iter().take(1))
    {
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for id in ids.iter().flatten() {
            let row = &mut expected[(id % 3) as usize];
            row.1 += 1;
            row.2 += id;
        }
    }
    assert_eq!(rows, expected);
}

/// A session dropped mid-write, without `abort`, has its open file's upload
/// aborted by the file's own drop, on the runtime in the background: a drop
/// cannot await. Nothing is left on disk once it has run.
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_session_aborts_its_open_upload_in_the_background() {
    let lake = Lake::new(None).await;
    let mut session = lake
        .table_writer(Staging::Parts {
            part_bytes: PART_BYTES,
        })
        .begin_write("main", "t", &schema(), WriteMode::Create)
        .unwrap();
    for batch in batches(5) {
        session.write_batch_async(&batch).await.unwrap();
    }
    let (opened, completed, _) = counts(&lake.store.events());
    assert!(opened > completed, "a file is open with parts sent");
    drop(session);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let events = lake.store.events();
        let (opened, completed, aborted) = counts(&events);
        if opened == completed + aborted && lake.parquet_on_disk().is_empty() {
            assert!(aborted >= 1, "{events:?}");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the dropped upload was never aborted: {events:?}, on disk {:?}",
            lake.parquet_on_disk()
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}
