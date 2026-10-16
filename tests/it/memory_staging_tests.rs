//! `Staging::Memory` — the only staging on `wasm32`, where there is no filesystem
//! — writes, deletes and reads back exactly what the default temp-file staging
//! does, and uploads without a tokio runtime to spawn on.
//!
//! Run natively, against an in-memory object store, so the browser's write path
//! is exercised in the same harness as everything else. The catalog is SQLite.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter, Staging, WriteMode,
};

fn db_url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

/// `count` batches of ascending ids, uneven in size so the files differ.
fn batches(count: i32) -> Vec<RecordBatch> {
    let mut next = 0;
    (0..count)
        .map(|b| {
            let rows = 40 + (b % 7) * 37;
            let ids: Vec<i32> = (next..next + rows).collect();
            next += rows;
            let vals: Vec<i32> = ids.iter().map(|id| id * 10).collect();
            RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
            )
            .unwrap()
        })
        .collect()
}

/// Small row groups and a 4 KiB target, so a few thousand rows roll into many
/// files; nothing inlined, so every row is in Parquet; and the given staging.
fn options(staging: Staging) -> DuckLakeWriteOptions {
    let mut options = DuckLakeWriteOptions::default();
    options.max_row_group_rows = Some(64);
    options.target_file_size = Some(4 * 1024);
    options.parquet_version = Some(parquet::file::properties::WriterVersion::PARQUET_1_0);
    options.data_inlining_row_limit = Some(0);
    options.staging = Some(staging);
    options
}

/// A SQLite catalog whose data path is `<temp>/data`, every byte of which lives in
/// `store`.
struct Lake {
    temp: TempDir,
    writer: Arc<SqliteMetadataWriter>,
    store: Arc<InMemory>,
}

impl Lake {
    async fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&db_url(&temp))
            .await
            .unwrap()
            .with_data_inlining_row_limit(0);
        writer
            .set_data_path(temp.path().join("data").to_str().unwrap())
            .unwrap();
        Self {
            temp,
            writer: Arc::new(writer),
            store: Arc::new(InMemory::new()),
        }
    }

    fn table_writer(&self, staging: Staging) -> DuckLakeTableWriter {
        let store: Arc<dyn ObjectStore> = self.store.clone();
        DuckLakeTableWriter::new(self.writer.clone(), store)
            .unwrap()
            .with_options(&options(staging))
    }

    /// A session whose DuckLake catalog writes with `staging`, its files in `store`.
    async fn session(&self, staging: Staging) -> SessionContext {
        let provider = SqliteMetadataProvider::new(&db_url(&self.temp))
            .await
            .unwrap();
        let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), self.writer.clone())
            .unwrap()
            .with_write_options(options(staging));
        let ctx = SessionContext::new();
        ctx.runtime_env().register_object_store(
            &url::Url::parse("file:///").unwrap(),
            self.store.clone() as Arc<dyn ObjectStore>,
        );
        ctx.register_catalog("ducklake", Arc::new(catalog));
        ctx
    }

    async fn sql(&self, ctx: &SessionContext, sql: &str) {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }

    async fn rows(&self, ctx: &SessionContext) -> Vec<(i32, i32)> {
        let batches = ctx
            .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for batch in batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let vals = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                rows.push((ids.value(i), vals.value(i)));
            }
        }
        rows
    }

    /// Every live data and delete file the catalog records, with the size and
    /// footer size it records, checked against the object actually stored.
    async fn live_files(&self) -> (usize, usize) {
        let pool = SqlitePool::connect(&db_url(&self.temp)).await.unwrap();
        let mut counts = (0, 0);
        for (table, sql, count) in [
            (
                "data",
                "SELECT path, file_size_bytes, footer_size FROM ducklake_data_file \
                 WHERE end_snapshot IS NULL",
                &mut counts.0,
            ),
            (
                "delete",
                "SELECT path, file_size_bytes, footer_size FROM ducklake_delete_file \
                 WHERE end_snapshot IS NULL",
                &mut counts.1,
            ),
        ] {
            let files: Vec<(String, i64, Option<i64>)> =
                sqlx::query_as(sql).fetch_all(&pool).await.unwrap();
            for (path, size, footer) in files {
                let name = path.rsplit('/').next().unwrap();
                let object = self.object_named(name).await;
                let bytes = self
                    .store
                    .get(&object)
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();
                assert_eq!(bytes.len() as i64, size, "{table} {path}: recorded size");
                let tail = &bytes[bytes.len() - 8..];
                assert_eq!(&tail[4..], b"PAR1", "{table} {path}: a whole Parquet file");
                let metadata = u32::from_le_bytes(tail[..4].try_into().unwrap());
                assert_eq!(
                    footer,
                    Some(i64::from(metadata)),
                    "{table} {path}: footer size"
                );
                *count += 1;
            }
        }
        counts
    }

    async fn object_named(&self, name: &str) -> ObjectPath {
        let objects: Vec<_> = self.store.list(None).try_collect().await.unwrap();
        objects
            .into_iter()
            .map(|meta| meta.location)
            .find(|location| location.filename() == Some(name))
            .unwrap_or_else(|| panic!("no object named {name}"))
    }
}

/// The same table written, deleted from, appended to and read back under each
/// staging: the same rows, the same number of files, every one recorded with its
/// true size and footer.
#[tokio::test(flavor = "multi_thread")]
async fn memory_staging_writes_deletes_and_reads_what_file_staging_does() {
    let mut outcomes = Vec::new();
    for staging in [Staging::File, Staging::Memory] {
        let lake = Lake::new().await;
        lake.table_writer(staging)
            .write_table("main", "t", &batches(30))
            .await
            .unwrap();
        let ctx = lake.session(staging).await;
        lake.sql(
            &ctx,
            "DELETE FROM ducklake.main.t WHERE id % 3 = 0 OR id BETWEEN 500 AND 900",
        )
        .await;
        lake.sql(
            &ctx,
            "INSERT INTO ducklake.main.t VALUES (100000, 1), (100001, 2)",
        )
        .await;
        let rows = lake.rows(&ctx).await;
        let files = lake.live_files().await;
        assert!(
            files.0 > 5,
            "{staging:?}: the write rolled into {} files",
            files.0
        );
        assert!(files.1 > 0, "{staging:?}: the delete wrote delete files");
        outcomes.push((rows, files));
    }
    let (file_rows, file_files) = &outcomes[0];
    let (memory_rows, memory_files) = &outcomes[1];

    let total: i32 = (0..30).map(|b| 40 + (b % 7) * 37).sum();
    let mut expected: Vec<(i32, i32)> = (0..total)
        .filter(|id| id % 3 != 0 && !(500..=900).contains(id))
        .map(|id| (id, id * 10))
        .collect();
    expected.extend([(100000, 1), (100001, 2)]);
    assert_eq!(file_rows, &expected, "file staging");
    assert_eq!(memory_rows, &expected, "memory staging");
    assert_eq!(
        memory_files, file_files,
        "the same files, staged either way"
    );
}

/// A streaming write driven with no tokio runtime on the thread — as a browser
/// host drives it — uploads every rolled file itself and commits them all. With a
/// runtime the uploads are spawned; without one, spawning would panic, and before
/// this the write failed at `finish` with "uploading data files requires a tokio
/// runtime".
#[tokio::test(flavor = "multi_thread")]
async fn a_write_with_no_tokio_runtime_uploads_its_files_itself() {
    for sync_feed in [true, false] {
        let lake = Lake::new().await;
        let table_writer = lake.table_writer(Staging::Memory);
        let input = batches(30);
        let committed = std::thread::spawn(move || {
            assert!(tokio::runtime::Handle::try_current().is_err());
            futures::executor::block_on(async {
                let mut session = table_writer
                    .begin_write("main", "t", &schema(), WriteMode::Replace)
                    .unwrap();
                for batch in &input {
                    if sync_feed {
                        session.write_batch(batch).unwrap();
                    } else {
                        session.write_batch_async(batch).await.unwrap();
                    }
                }
                session.finish().await.unwrap().records_written
            })
        })
        .join()
        .unwrap();

        let total: i32 = (0..30).map(|b| 40 + (b % 7) * 37).sum();
        assert_eq!(committed, i64::from(total), "sync_feed={sync_feed}");
        let ctx = lake.session(Staging::Memory).await;
        let rows = lake.rows(&ctx).await;
        let expected: Vec<(i32, i32)> = (0..total).map(|id| (id, id * 10)).collect();
        assert_eq!(rows, expected, "sync_feed={sync_feed}");
        let (data, _) = lake.live_files().await;
        assert!(data > 5, "sync_feed={sync_feed}: rolled into {data} files");
    }
}
