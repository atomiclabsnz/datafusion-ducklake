//! The crate writes into a Postgres catalog the DuckDB extension created, and the extension
//! goes on reading and writing it — the Postgres counterpart of
//! `duckdb_created_sqlite_catalog_tests`.
//!
//! The extension's Postgres schema has no IDENTITY columns and no keys this crate's own
//! schema relies on, keeps its id allocators (`next_catalog_id`, `next_file_id`) on every
//! snapshot row, and allocates snapshot ids itself. Only the extension, attached after the
//! crate wrote, catches a write that misses any of that.
//!
//! Runs against `DUCKLAKE_TEST_POSTGRES_URL` (a server URL; each test creates its own
//! database there) when it is set, otherwise against a Postgres container.

#![cfg(all(feature = "write-postgres", feature = "metadata-duckdb"))]

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeTableWriter, MetadataWriter, PostgresSingleCatalogMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

/// A fresh, empty database: its `postgresql://` URL, its libpq connection string for the
/// extension, and whatever must stay alive for it to.
struct Database {
    url: String,
    libpq: String,
    admin: String,
    name: String,
    _container: Option<ContainerAsync<Postgres>>,
}

/// Dropped with the test, so runs against a shared server do not accumulate databases.
impl Drop for Database {
    fn drop(&mut self) {
        let (admin, name) = (self.admin.clone(), self.name.clone());
        let _ = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                if let Ok(pool) = sqlx::PgPool::connect(&admin).await {
                    let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                        "DROP DATABASE IF EXISTS {name} WITH (FORCE)"
                    )))
                    .execute(&pool)
                    .await;
                }
            });
        })
        .join();
    }
}

async fn fresh_database() -> Database {
    let (server, container) = match std::env::var("DUCKLAKE_TEST_POSTGRES_URL") {
        Ok(server) => (url::Url::parse(&server).unwrap(), None),
        Err(_) => {
            let container = Postgres::default().start().await.unwrap();
            let port = container.get_host_port_ipv4(5432).await.unwrap();
            let server = url::Url::parse(&format!(
                "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
            ))
            .unwrap();
            (server, Some(container))
        },
    };
    let name = format!("ducklake_{}", uuid::Uuid::new_v4().simple());
    let admin = sqlx::PgPool::connect(server.as_str()).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {name}")))
        .execute(&admin)
        .await
        .unwrap();
    admin.close().await;
    let mut url = server.clone();
    url.set_path(&name);
    let libpq = format!(
        "dbname={name} host={} port={} user={} password={}",
        server.host_str().unwrap(),
        server.port().unwrap_or(5432),
        server.username(),
        server.password().unwrap_or_default()
    );
    Database {
        url: url.to_string(),
        libpq,
        admin: server.to_string(),
        name,
        _container: container,
    }
}

static INSTALL_EXTENSIONS: std::sync::Once = std::sync::Once::new();

fn attach(
    db: &Database,
    data_path: &std::path::Path,
    inlining_row_limit: usize,
) -> duckdb::Connection {
    INSTALL_EXTENSIONS.call_once(|| {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch("INSTALL ducklake; INSTALL postgres;")
            .unwrap();
    });
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD postgres;").unwrap();
    std::fs::create_dir_all(data_path).unwrap();
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:postgres:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT {})",
        db.libpq,
        data_path.display(),
        inlining_row_limit
    ))
    .unwrap();
    conn
}

fn orders(ids: &[i64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("amount", DataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(Float64Array::from(
                ids.iter().map(|id| *id as f64).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn found(conn: &duckdb::Connection, sql: &str) -> (i64, i64) {
    conn.query_row(sql, [], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
}

async fn ids(pool: &sqlx::PgPool, columns: &[(&str, &str)]) -> Vec<i64> {
    let mut ids = Vec::new();
    for (table, column) in columns {
        let rows: Vec<i64> =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT {column} FROM {table}")))
                .fetch_all(pool)
                .await
                .unwrap();
        ids.extend(rows);
    }
    ids
}

#[tokio::test(flavor = "multi_thread")]
async fn the_extension_reads_and_writes_a_postgres_catalog_after_the_crate_wrote_to_it() {
    let db = fresh_database().await;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    {
        // Files, two deletes, a view, and a table of inlined rows — all the extension's.
        let lake = attach(&db, &data, 0);
        lake.execute_batch(
            "CREATE TABLE lake.main.orders (id BIGINT, amount DOUBLE);
             INSERT INTO lake.main.orders SELECT i, i FROM range(1, 11) r(i);
             DELETE FROM lake.main.orders WHERE id <= 2;
             DELETE FROM lake.main.orders WHERE id = 10;
             CREATE VIEW lake.main.big_orders AS SELECT * FROM lake.main.orders WHERE id > 5;",
        )
        .unwrap();
        let lake = attach(&db, &data, 100);
        lake.execute_batch(
            "CREATE TABLE lake.main.small (id BIGINT, amount DOUBLE);
             INSERT INTO lake.main.small VALUES (1, 1), (2, 2);",
        )
        .unwrap();
    }

    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&db.url)
            .await
            .unwrap(),
    );
    let tables =
        DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
    tables
        .append_table("main", "orders", &[orders(&[11, 12, 13])])
        .await
        .unwrap();
    // A file beside the extension's inlined rows.
    tables
        .append_table("main", "small", &[orders(&[100])])
        .await
        .unwrap();
    tables
        .write_table("main", "fresh", &[orders(&[100, 200])])
        .await
        .unwrap();
    drop(tables);
    drop(writer);

    {
        let lake = attach(&db, &data, 100);
        let q = |sql: &str| found(&lake, sql);
        // 3..=9 the extension left, 11..=13 the crate appended.
        assert_eq!(
            q("SELECT count(*), sum(id)::BIGINT FROM lake.main.orders"),
            (10, 78)
        );
        assert_eq!(
            q("SELECT count(*), sum(id)::BIGINT FROM lake.main.fresh"),
            (2, 300)
        );
        for id in [1, 2, 100] {
            assert_eq!(
                q(&format!(
                    "SELECT count(*), 0 FROM lake.main.small WHERE id = {id}"
                )),
                (1, 0),
                "a filter on small.id = {id} finds its row"
            );
        }
        // The extension allocates after the crate: a file, a delete file, a table.
        lake.execute_batch(
            "INSERT INTO lake.main.orders VALUES (14, 14);
             DELETE FROM lake.main.orders WHERE id = 11;
             CREATE TABLE lake.main.later AS SELECT 1::BIGINT AS id;
             INSERT INTO lake.main.small VALUES (3, 3);",
        )
        .unwrap();
        assert_eq!(
            q("SELECT count(*), sum(id)::BIGINT FROM lake.main.orders"),
            (10, 81)
        );
        assert_eq!(
            q("SELECT count(*), sum(id)::BIGINT FROM lake.main.later"),
            (1, 1)
        );
        assert_eq!(q("SELECT count(*), 0 FROM lake.main.big_orders"), (7, 0));
        assert_eq!(
            q("SELECT count(*), count(DISTINCT rowid) FROM lake.main.small"),
            (4, 4),
            "every row has a row id of its own"
        );
    }

    // The crate writes again after the extension: its snapshot counter has not seen the
    // extension's snapshots, so a snapshot id from it alone would collide.
    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&db.url)
            .await
            .unwrap(),
    );
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .append_table("main", "orders", &[orders(&[15])])
        .await
        .unwrap();
    {
        let lake = attach(&db, &data, 100);
        assert_eq!(
            found(
                &lake,
                "SELECT count(*), sum(id)::BIGINT FROM lake.main.orders"
            ),
            (11, 96)
        );
        lake.execute_batch("INSERT INTO lake.main.orders VALUES (16, 16);")
            .unwrap();
        assert_eq!(
            found(
                &lake,
                "SELECT count(*), sum(id)::BIGINT FROM lake.main.orders"
            ),
            (12, 112)
        );
    }

    // A replace over the extension's inlined rows retires them as well as files.
    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&db.url)
            .await
            .unwrap(),
    );
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "small", &[orders(&[50])])
        .await
        .unwrap();
    {
        let lake = attach(&db, &data, 100);
        assert_eq!(
            found(
                &lake,
                "SELECT count(*), sum(id)::BIGINT FROM lake.main.small"
            ),
            (1, 50)
        );
    }

    let pool = sqlx::PgPool::connect(&db.url).await.unwrap();
    for namespace in [
        &[
            ("ducklake_schema", "schema_id"),
            ("ducklake_table", "table_id"),
            ("ducklake_view", "view_id"),
        ][..],
        &[("ducklake_data_file", "data_file_id"), ("ducklake_delete_file", "delete_file_id")][..],
    ] {
        let ids = ids(&pool, namespace).await;
        let distinct: HashSet<_> = ids.iter().collect();
        assert_eq!(
            distinct.len(),
            ids.len(),
            "{namespace:?} reuses an id: {ids:?}"
        );
    }
    let unset: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_snapshot
         WHERE next_catalog_id IS NULL OR next_file_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(unset, 0, "a snapshot left the extension's allocators NULL");
    let stats_rows: Vec<(i64, i64)> =
        sqlx::query_as("SELECT table_id, count(*) FROM ducklake_table_stats GROUP BY table_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert!(
        stats_rows.iter().all(|(_, n)| *n == 1),
        "one ducklake_table_stats row per table: {stats_rows:?}"
    );
}

/// The writer names the snapshot its latest write committed, for a caller whose write went
/// through DataFusion and so got back only a row count.
#[tokio::test(flavor = "multi_thread")]
async fn the_postgres_writer_names_the_snapshot_it_committed() {
    let db = fresh_database().await;
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    {
        let lake = attach(&db, &data, 0);
        lake.execute_batch("CREATE TABLE lake.main.orders (id BIGINT, amount DOUBLE);")
            .unwrap();
    }
    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&db.url)
            .await
            .unwrap(),
    );
    assert_eq!(writer.last_committed_snapshot(), None);
    for _ in 0..2 {
        let written = DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new()))
            .unwrap()
            .append_table("main", "orders", &[orders(&[1, 2])])
            .await
            .unwrap();
        assert_eq!(writer.last_committed_snapshot(), Some(written.snapshot_id));
    }
}

/// A DuckDB-created lake with `CREATE TABLE lake.main.<table> (...)` run first, and a
/// writer on it.
async fn lake_with(
    ddl: &str,
    inlining_row_limit: usize,
) -> (Database, TempDir, Arc<PostgresSingleCatalogMetadataWriter>) {
    let db = fresh_database().await;
    let temp = TempDir::new().unwrap();
    attach(&db, &temp.path().join("data"), inlining_row_limit)
        .execute_batch(ddl)
        .unwrap();
    let writer = Arc::new(
        PostgresSingleCatalogMetadataWriter::new_with_init(&db.url)
            .await
            .unwrap(),
    );
    (db, temp, writer)
}

/// One transaction creating two new tables: each gets an id of its own, though no id is
/// reserved before commit in such a catalog.
#[tokio::test(flavor = "multi_thread")]
async fn one_transaction_creates_two_tables() {
    let (db, temp, writer) = lake_with(
        "CREATE TABLE lake.main.orders (id BIGINT, amount DOUBLE);",
        0,
    )
    .await;
    let tables = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
    let batch = orders(&[1, 2]);
    let mut txn = tables.transaction();
    for table in ["a", "b"] {
        txn.stage_write(
            "main",
            table,
            &batch.schema(),
            WriteMode::Append,
            std::slice::from_ref(&batch),
        )
        .await
        .unwrap();
    }
    txn.commit().await.unwrap();
    let lake = attach(&db, &temp.path().join("data"), 0);
    for table in ["a", "b"] {
        assert_eq!(
            found(
                &lake,
                &format!("SELECT count(*), sum(id)::BIGINT FROM lake.main.{table}")
            ),
            (2, 3)
        );
    }
}

/// A replace whose base is older than rows the extension inlined since is a conflict, as
/// it is over a file the extension wrote since — not a silent retirement of those rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_conflicts_with_rows_inlined_since_its_base() {
    for inlining in [100, 0] {
        let (db, temp, writer) = lake_with(
            "CREATE TABLE lake.main.small (id BIGINT, amount DOUBLE);
             INSERT INTO lake.main.small VALUES (1, 1), (2, 2);",
            100,
        )
        .await;
        let data = temp.path().join("data");
        let tables = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
        let batch = orders(&[50]);
        let mut txn = tables.transaction();
        txn.stage_write(
            "main",
            "small",
            &batch.schema(),
            WriteMode::Replace,
            std::slice::from_ref(&batch),
        )
        .await
        .unwrap();
        attach(&db, &data, inlining)
            .execute_batch("INSERT INTO lake.main.small VALUES (3, 3);")
            .unwrap();
        let committed = txn.commit().await;
        assert!(
            matches!(
                committed,
                Err(datafusion_ducklake::DuckLakeError::Conflict(_))
            ),
            "inlining {inlining}: {committed:?}"
        );
        assert_eq!(
            found(
                &attach(&db, &data, 100),
                "SELECT count(*), sum(id)::BIGINT FROM lake.main.small"
            ),
            (3, 6),
            "inlining {inlining}: the extension's rows survive"
        );
    }
}

/// A table id is allocated against committed state: a concurrent writer's uncommitted
/// table is waited for, not handed out again — the extension's `ducklake_table` has no
/// key that would stop the second.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_id_is_not_handed_out_twice() {
    let (db, _temp, writer) = lake_with(
        "CREATE TABLE lake.main.orders (id BIGINT, amount DOUBLE);",
        0,
    )
    .await;
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    // Another writer, mid-transaction, has taken the next free id for a table of its own.
    let pool = sqlx::PgPool::connect(&db.url).await.unwrap();
    let mut other = pool.begin().await.unwrap();
    let next: i64 = sqlx::query_scalar("SELECT MAX(next_catalog_id) FROM ducklake_snapshot")
        .fetch_one(&mut *other)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_table
             (table_id, table_uuid, begin_snapshot, schema_id, table_name, path, path_is_relative)
         VALUES ($1, gen_random_uuid(), $2, $3, 'other', 'other/', true)",
    )
    .bind(next)
    .bind(snapshot)
    .bind(schema_id)
    .execute(&mut *other)
    .await
    .unwrap();
    let w = writer.clone();
    let mine = tokio::task::spawn_blocking(move || {
        w.get_or_create_table(schema_id, "mine", None, snapshot)
    });
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    other.commit().await.unwrap();
    let (mine, created) = mine.await.unwrap().unwrap();
    assert!(created);
    assert_ne!(mine, next, "the id the other writer took");
    let duplicates: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM (SELECT table_id FROM ducklake_table
         GROUP BY table_id HAVING count(*) > 1) d",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(duplicates, 0);
}

fn wide(names: &[&str], values: &[i64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|n| Field::new(*n, DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(
        schema,
        values
            .iter()
            .map(|v| Arc::new(Int64Array::from(vec![*v])) as arrow::array::ArrayRef)
            .collect(),
    )
    .unwrap()
}

/// A column the crate adds after the extension added one takes an id of its own: the
/// extension numbers a table's columns from that table's highest id, past the crate's
/// counter, and a shared id would read one column's values under the other's name.
#[tokio::test(flavor = "multi_thread")]
async fn a_column_added_after_the_extension_added_one_has_its_own_id() {
    let (db, temp, writer) = lake_with("CREATE TABLE lake.main.t (a BIGINT, b BIGINT);", 0).await;
    let data = temp.path().join("data");
    let tables = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
    tables
        .append_table("main", "t", &[wide(&["a", "b", "c"], &[1, 1, 1])])
        .await
        .unwrap();
    attach(&db, &data, 0)
        .execute_batch("ALTER TABLE lake.main.t ADD COLUMN d BIGINT; INSERT INTO lake.main.t VALUES (2, 2, 2, 2);")
        .unwrap();
    tables
        .append_table(
            "main",
            "t",
            &[wide(&["e", "a", "b", "c", "d"], &[99, 3, 3, 3, 3])],
        )
        .await
        .unwrap();
    assert_eq!(
        found(
            &attach(&db, &data, 0),
            "SELECT d, e FROM lake.main.t WHERE a = 3"
        ),
        (3, 99)
    );
}
