#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]
//! A replace ends the live delete files of the data files it ends, in the same commit, as
//! official DuckLake's `CREATE OR REPLACE` does. Left live, the catalog would list a delete
//! file whose data file is dead. Each backend ends them with its own SQL, so each runs it;
//! `end_table_files` is checked beside the replace.

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, MetadataWriter,
    SqliteMetadataProvider, SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

/// Live delete files whose data file has ended.
const LIVE_DELETES_OF_ENDED: &str = "SELECT COUNT(*) FROM ducklake_delete_file del
     JOIN ducklake_data_file d ON d.data_file_id = del.data_file_id
     WHERE del.end_snapshot IS NULL AND d.end_snapshot IS NOT NULL";

/// Delete files that ended in the snapshot their data file ended in.
const DELETES_ENDED_WITH_DATA: &str = "SELECT COUNT(*) FROM ducklake_delete_file del
     JOIN ducklake_data_file d ON d.data_file_id = del.data_file_id
     WHERE del.end_snapshot = d.end_snapshot";

/// A live delete file on each live data file, as another writer's `DELETE` (official
/// DuckLake's) would record it. Only the catalog row: nothing here reads the file.
/// `{id}` is the id column and its value, for a backend that assigns no id itself.
const SEED_DELETES: &str = "INSERT INTO ducklake_delete_file
     ({id_column}data_file_id, table_id, path, path_is_relative, file_size_bytes,
      delete_count, begin_snapshot)
     SELECT {id_value}data_file_id, table_id, 'seeded-delete.parquet', TRUE, 1, 1, begin_snapshot
     FROM ducklake_data_file WHERE end_snapshot IS NULL";

/// [`SEED_DELETES`] for a backend that assigns delete file ids, or (`explicit_id`) not.
fn seed_deletes(explicit_id: bool) -> String {
    let (column, value) = if explicit_id {
        ("delete_file_id, ", "1000 + data_file_id, ")
    } else {
        ("", "")
    };
    SEED_DELETES
        .replace("{id_column}", column)
        .replace("{id_value}", value)
}

/// `main.t`'s live table id and the catalog head.
const TABLE_AND_HEAD: [&str; 2] = [
    "SELECT MAX(table_id) FROM ducklake_table
     WHERE table_name = 't' AND end_snapshot IS NULL",
    "SELECT MAX(snapshot_id) FROM ducklake_snapshot",
];

fn batch(ids: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(ids))],
    )
    .unwrap()
}

fn data_path(temp: &TempDir) -> String {
    let path = temp.path().join("data");
    std::fs::create_dir_all(&path).unwrap();
    path.to_str().unwrap().to_string()
}

/// On a backend reached through `sql` (a statement, run) and `count` (a query, its one
/// integer): a delete file on each of `t`'s two data files, then a replace of `t`. The
/// replace ends both delete files with their data files; then `end_table_files` does the
/// same for the replace's own file.
async fn replace_ends_delete_files<SqlFut, CountFut>(
    metadata: Arc<dyn MetadataWriter>,
    explicit_id: bool,
    sql: impl Fn(String) -> SqlFut,
    count: impl Fn(&'static str) -> CountFut,
) where
    SqlFut: std::future::Future<Output = ()>,
    CountFut: std::future::Future<Output = i64>,
{
    let writer =
        DuckLakeTableWriter::new(Arc::clone(&metadata), Arc::new(LocalFileSystem::new())).unwrap();
    writer
        .write_table("main", "t", &[batch(vec![1, 2])])
        .await
        .unwrap();
    writer
        .append_table("main", "t", &[batch(vec![3])])
        .await
        .unwrap();
    sql(seed_deletes(explicit_id)).await;

    writer
        .write_table("main", "t", &[batch(vec![9])])
        .await
        .unwrap();
    assert_eq!(
        count(LIVE_DELETES_OF_ENDED).await,
        0,
        "a replace leaves no live delete file on a data file it ended"
    );
    assert_eq!(
        count(DELETES_ENDED_WITH_DATA).await,
        2,
        "each delete file ends in the replace's snapshot"
    );

    sql(seed_deletes(explicit_id)).await;
    let table_id = count(TABLE_AND_HEAD[0]).await;
    let head = count(TABLE_AND_HEAD[1]).await;
    assert_eq!(metadata.end_table_files(table_id, head + 1).unwrap(), 1);
    assert_eq!(
        count(LIVE_DELETES_OF_ENDED).await,
        0,
        "end_table_files leaves no live delete file on a data file it ended"
    );
    assert_eq!(count(DELETES_ENDED_WITH_DATA).await, 3);
}

async fn sqlite_writer(temp: &TempDir) -> SqliteMetadataWriter {
    let conn = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.db").display()
    );
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(&data_path(temp)).unwrap();
    writer.with_data_inlining_row_limit(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_replace_ends_delete_files_on_sqlite() {
    let temp = TempDir::new().unwrap();
    let metadata: Arc<dyn MetadataWriter> = Arc::new(sqlite_writer(&temp).await);
    let pool = sqlx::SqlitePool::connect(&format!(
        "sqlite:{}",
        temp.path().join("catalog.db").display()
    ))
    .await
    .unwrap();
    replace_ends_delete_files(
        metadata,
        false,
        |statement| {
            let pool = pool.clone();
            async move {
                sqlx::query(sqlx::AssertSqlSafe(statement))
                    .execute(&pool)
                    .await
                    .unwrap();
            }
        },
        |query| {
            let pool = pool.clone();
            async move { sqlx::query_scalar(query).fetch_one(&pool).await.unwrap() }
        },
    )
    .await;
}

/// The issue's case through SQL: a keyed write (`DELETE`, then `INSERT`) leaves a data
/// file, its delete file and a second data file; `INSERT OVERWRITE` then ends all three.
#[tokio::test(flavor = "multi_thread")]
async fn insert_overwrite_after_a_keyed_write_ends_its_delete_file() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(sqlite_writer(&temp).await);
    DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3])])
        .await
        .unwrap();
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer)
        .unwrap()
        .with_write_options(DuckLakeWriteOptions::default().with_data_inlining_row_limit(0));
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    for statement in ["DELETE FROM lake.main.t WHERE id = 2", "INSERT INTO lake.main.t VALUES (2)"]
    {
        ctx.sql(statement).await.unwrap().collect().await.unwrap();
    }
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    let live_deletes: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(live_deletes, 1, "the keyed write leaves a live delete file");

    ctx.sql("INSERT OVERWRITE lake.main.t VALUES (7)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let orphans: i64 = sqlx::query_scalar(LIVE_DELETES_OF_ENDED)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        orphans, 0,
        "the overwrite ends the delete file with its data file"
    );
    let ended_with_data: i64 = sqlx::query_scalar(DELETES_ENDED_WITH_DATA)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ended_with_data, 1);
    let rows = ctx
        .sql("SELECT id FROM lake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_ends_delete_files_on_duckdb() {
    // DuckDB allows one client per catalog file, so each statement opens the file while
    // the writer has it closed: the writer is reopened for every step.
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.duckdb");
    let path = path.to_str().unwrap().to_string();
    let writer = datafusion_ducklake::DuckdbMetadataWriter::new_with_init(&path).unwrap();
    writer.set_data_path(&data_path(&temp)).unwrap();
    drop(writer);
    let open = || -> DuckLakeTableWriter {
        let writer = datafusion_ducklake::DuckdbMetadataWriter::new(&path)
            .unwrap()
            .with_data_inlining_row_limit(0);
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap()
    };
    let sql = |statement: &str| {
        duckdb::Connection::open(&path)
            .unwrap()
            .execute(statement, [])
            .unwrap();
    };
    let count = |query: &str| -> i64 {
        duckdb::Connection::open(&path)
            .unwrap()
            .query_row(query, [], |row| row.get(0))
            .unwrap()
    };

    let writer = open();
    writer
        .write_table("main", "t", &[batch(vec![1, 2])])
        .await
        .unwrap();
    writer
        .append_table("main", "t", &[batch(vec![3])])
        .await
        .unwrap();
    drop(writer);
    sql(&seed_deletes(true));

    let writer = open();
    writer
        .write_table("main", "t", &[batch(vec![9])])
        .await
        .unwrap();
    drop(writer);
    assert_eq!(count(LIVE_DELETES_OF_ENDED), 0);
    assert_eq!(count(DELETES_ENDED_WITH_DATA), 2);

    sql(&seed_deletes(true));
    let table_id = count(TABLE_AND_HEAD[0]);
    let head = count(TABLE_AND_HEAD[1]);
    let metadata = datafusion_ducklake::DuckdbMetadataWriter::new(&path).unwrap();
    assert_eq!(metadata.end_table_files(table_id, head + 1).unwrap(), 1);
    drop(metadata);
    assert_eq!(count(LIVE_DELETES_OF_ENDED), 0);
    assert_eq!(count(DELETES_ENDED_WITH_DATA), 3);
}

/// Where a test container's mapped ports answer: `TESTCONTAINERS_HOST_OVERRIDE` when set,
/// else what testcontainers says.
#[cfg(any(feature = "write-mysql", feature = "write-postgres"))]
async fn container_host<I: testcontainers::Image>(
    container: &testcontainers::ContainerAsync<I>,
) -> String {
    match std::env::var("TESTCONTAINERS_HOST_OVERRIDE") {
        Ok(host) if !host.is_empty() => host,
        _ => container.get_host().await.unwrap().to_string(),
    }
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_ends_delete_files_on_mysql() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let host = container_host(&container).await;
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let url = format!("mysql://root@{host}:{port}/test");
    let writer = datafusion_ducklake::MySqlMetadataWriter::new_with_init(&url)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    writer.set_data_path(&data_path(&temp)).unwrap();
    let pool = sqlx::MySqlPool::connect(&url).await.unwrap();
    replace_ends_delete_files(
        Arc::new(writer.with_data_inlining_row_limit(0)),
        false,
        |statement| {
            let pool = pool.clone();
            async move {
                sqlx::query(sqlx::AssertSqlSafe(statement))
                    .execute(&pool)
                    .await
                    .unwrap();
            }
        },
        |query| {
            let pool = pool.clone();
            async move { sqlx::query_scalar(query).fetch_one(&pool).await.unwrap() }
        },
    )
    .await;
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_ends_delete_files_on_both_postgres_writers() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let host = container_host(&container).await;
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let base = format!("postgresql://postgres:postgres@{host}:{port}");
    let admin = sqlx::PgPool::connect(&format!("{base}/postgres"))
        .await
        .unwrap();
    for database in ["single", "multi"] {
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
    }
    let run = |pool: sqlx::PgPool| {
        move |statement: String| {
            let pool = pool.clone();
            async move {
                sqlx::query(sqlx::AssertSqlSafe(statement))
                    .execute(&pool)
                    .await
                    .unwrap();
            }
        }
    };
    let count = |pool: sqlx::PgPool| {
        move |query: &'static str| {
            let pool = pool.clone();
            async move { sqlx::query_scalar(query).fetch_one(&pool).await.unwrap() }
        }
    };

    let temp = TempDir::new().unwrap();
    let single = datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&format!(
        "{base}/single"
    ))
    .await
    .unwrap();
    single.set_data_path(&data_path(&temp)).unwrap();
    let pool = sqlx::PgPool::connect(&format!("{base}/single"))
        .await
        .unwrap();
    replace_ends_delete_files(
        Arc::new(single.with_data_inlining_row_limit(0)),
        false,
        run(pool.clone()),
        count(pool),
    )
    .await;

    let pool = sqlx::PgPool::connect(&format!("{base}/multi"))
        .await
        .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    let multi = datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    multi.set_data_path(&data_path(&temp)).unwrap();
    replace_ends_delete_files(
        Arc::new(multi.with_data_inlining_row_limit(0)),
        false,
        run(pool.clone()),
        count(pool),
    )
    .await;
}
