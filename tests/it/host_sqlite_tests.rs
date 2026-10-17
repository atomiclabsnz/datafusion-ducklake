//! The SQLite catalog over a host's SQLite (`host_sqlite`), with DuckDB as the oracle.
//!
//! The host here is [`SqlxHost`], a native SQLite connection behind the same synchronous
//! interface a browser page's SQLite is given through. What a tab writes, DuckDB must read back
//! unchanged, and what DuckDB wrote, the host backend must read and go on writing. The rules a
//! host on OPFS needs are asserted on the statements the host is asked to run.

#![cfg(all(
    feature = "host-sqlite",
    feature = "write-sqlite",
    feature = "metadata-sqlite",
    feature = "metadata-duckdb"
))]

use std::sync::{Arc, Mutex};

use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_ducklake::host_sqlite::{
    HostError, HostRows, HostSqlite, HostSqliteMetadataProvider, HostSqliteMetadataWriter,
    SqlxHost, Value,
};
use datafusion_ducklake::{DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, MetadataWriter};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

fn catalog_url(temp: &TempDir) -> String {
    format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.sqlite").display()
    )
}

fn data_dir(temp: &TempDir) -> std::path::PathBuf {
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    data
}

fn ids(values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

/// A context whose `ducklake` catalog reads and writes through `host`.
fn context(host: Arc<dyn HostSqlite>) -> SessionContext {
    let provider = HostSqliteMetadataProvider::from_host(Arc::clone(&host));
    let writer = HostSqliteMetadataWriter::from_host(host).unwrap();
    // Seeds the crate's own counters in a catalog DuckDB created; a no-op once they exist.
    writer.initialize_schema().unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

async fn read_ids(ctx: &SessionContext) -> Vec<i32> {
    let batches = ctx
        .sql("SELECT id FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..column.len())
                .map(|i| column.value(i))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn run(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// DuckDB with the lake attached; `DATA_INLINING_ROW_LIMIT 0` keeps its writes in files.
fn duckdb(temp: &TempDir) -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite;
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0);",
            temp.path().join("catalog.sqlite").display(),
            data_dir(temp).display()
        ))
        .unwrap();
    connection
}

fn duckdb_ids(connection: &duckdb::Connection) -> Vec<i32> {
    let mut statement = connection
        .prepare("SELECT id FROM lake.main.t ORDER BY id")
        .unwrap();
    statement
        .query_map([], |row| row.get::<_, i32>(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_reads_a_lake_written_through_the_host() {
    let temp = TempDir::new().unwrap();
    // DuckDB creates the empty catalog: one this crate initialises is not the schema DuckDB
    // reads (it has no `schema_uuid`, no id watermarks). Everything after is the host's.
    drop(duckdb(&temp));
    let host = SqlxHost::open(&catalog_url(&temp)).unwrap();
    let writer = HostSqliteMetadataWriter::from_host(Arc::clone(&host))
        .unwrap()
        .with_data_inlining_row_limit(0);
    writer.initialize_schema().unwrap();
    writer
        .set_data_path(data_dir(&temp).to_str().unwrap())
        .unwrap();
    let tables =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    tables
        .write_table("main", "t", &[ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])
        .await
        .unwrap();

    // A delete and an append through SQL, as a keyed write will issue them.
    let ctx = context(Arc::clone(&host));
    run(&ctx, "DELETE FROM ducklake.main.t WHERE id IN (2, 7)").await;
    run(&ctx, "INSERT INTO ducklake.main.t VALUES (11), (12)").await;
    let expected = vec![1, 3, 4, 5, 6, 8, 9, 10, 11, 12];
    assert_eq!(read_ids(&context(Arc::clone(&host))).await, expected);
    drop((ctx, tables, host));

    let lake = duckdb(&temp);
    assert_eq!(duckdb_ids(&lake), expected);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_host_reads_and_writes_a_lake_duckdb_wrote() {
    let temp = TempDir::new().unwrap();
    {
        let lake = duckdb(&temp);
        lake.execute_batch(
            "CREATE TABLE lake.main.t (id INTEGER);
             INSERT INTO lake.main.t SELECT range::INTEGER FROM range(1, 11);
             DELETE FROM lake.main.t WHERE id = 4;",
        )
        .unwrap();
    }

    let host = SqlxHost::open(&catalog_url(&temp)).unwrap();
    let ctx = context(Arc::clone(&host));
    assert_eq!(read_ids(&ctx).await, vec![1, 2, 3, 5, 6, 7, 8, 9, 10]);
    run(&ctx, "DELETE FROM ducklake.main.t WHERE id > 8").await;
    run(&ctx, "INSERT INTO ducklake.main.t VALUES (20)").await;
    let expected = vec![1, 2, 3, 5, 6, 7, 8, 20];
    assert_eq!(read_ids(&context(Arc::clone(&host))).await, expected);
    drop((ctx, host));

    let lake = duckdb(&temp);
    assert_eq!(duckdb_ids(&lake), expected);
    // And DuckDB goes on writing it.
    lake.execute_batch("INSERT INTO lake.main.t VALUES (30)")
        .unwrap();
    assert_eq!(duckdb_ids(&lake), [expected, vec![30]].concat());
}

/// The statements a host was asked to run, each with whether a transaction was open.
struct Recording {
    inner: Arc<dyn HostSqlite>,
    log: Mutex<Vec<(String, bool)>>,
    depth: Mutex<u32>,
}

impl Recording {
    fn new(inner: Arc<dyn HostSqlite>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            log: Mutex::new(Vec::new()),
            depth: Mutex::new(0),
        })
    }

    fn take(&self) -> Vec<(String, bool)> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl HostSqlite for Recording {
    fn query(&self, sql: &str, params: &[Value]) -> Result<HostRows, HostError> {
        let head = sql.trim_start().to_ascii_uppercase();
        let mut depth = self.depth.lock().unwrap();
        let result = self.inner.query(sql, params);
        if result.is_ok() {
            if head.starts_with("BEGIN") || head.starts_with("SAVEPOINT") {
                *depth += 1;
            } else if head.starts_with("COMMIT")
                || head.starts_with("ROLLBACK TO")
                || head.starts_with("RELEASE")
                || head.starts_with("ROLLBACK")
            {
                if !head.starts_with("ROLLBACK TO") {
                    *depth = depth.saturating_sub(1);
                }
            } else {
                self.log.lock().unwrap().push((sql.to_string(), *depth > 0));
            }
        }
        result
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_read_runs_every_statement_in_one_transaction() {
    let temp = TempDir::new().unwrap();
    let host = SqlxHost::open(&catalog_url(&temp)).unwrap();
    {
        let writer = HostSqliteMetadataWriter::from_host(Arc::clone(&host))
            .unwrap()
            .with_data_inlining_row_limit(0);
        writer.initialize_schema().unwrap();
        writer
            .set_data_path(data_dir(&temp).to_str().unwrap())
            .unwrap();
        let tables =
            DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
        tables
            .write_table("main", "t", &[ids(&[1, 2, 3])])
            .await
            .unwrap();
    }

    let recording = Recording::new(host);
    let provider =
        HostSqliteMetadataProvider::from_host(Arc::clone(&recording) as Arc<dyn HostSqlite>);
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "ducklake",
        Arc::new(DuckLakeCatalog::new(provider).unwrap()),
    );
    recording.take();
    assert_eq!(read_ids(&ctx).await, vec![1, 2, 3]);
    let statements = recording.take();
    assert!(!statements.is_empty());
    let outside: Vec<_> = statements.iter().filter(|(_, inside)| !inside).collect();
    assert!(
        outside.is_empty(),
        "read outside a transaction: {outside:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_while_another_connection_holds_the_catalog_is_refused_as_busy() {
    let temp = TempDir::new().unwrap();
    let host =
        SqlxHost::open_with_busy_timeout(&catalog_url(&temp), std::time::Duration::from_millis(50))
            .unwrap();
    let writer = HostSqliteMetadataWriter::from_host(Arc::clone(&host)).unwrap();
    writer.initialize_schema().unwrap();

    // Another connection takes the write lock and keeps it.
    let other = SqlxHost::open(&catalog_url(&temp)).unwrap();
    other.query("BEGIN IMMEDIATE", &[]).unwrap();

    let refused = writer.create_snapshot().unwrap_err();
    assert!(
        matches!(refused, DuckLakeError::CatalogBusy(_)),
        "expected CatalogBusy, got {refused:?}"
    );

    // Once the other connection lets go, the same call goes through.
    other.query("ROLLBACK", &[]).unwrap();
    writer.create_snapshot().unwrap();
}
