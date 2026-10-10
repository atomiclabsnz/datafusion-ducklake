//! The crate writes into a SQLite catalog the DuckDB extension created, and the extension goes on
//! reading and writing it.
//!
//! The extension stores its id allocators, `next_catalog_id` and `next_file_id`, on every
//! snapshot row and reads both at the head before any access. A snapshot the crate commits with
//! those NULL makes the catalog unreadable to the extension (#292), and an id the crate takes
//! below them is one the extension hands out again. Only the extension itself, attached after the
//! crate wrote, catches either.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{DuckLakeTableWriter, SqliteMetadataWriter};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

static INSTALL_EXTENSIONS: std::sync::Once = std::sync::Once::new();

fn attach(catalog: &std::path::Path) -> duckdb::Connection {
    INSTALL_EXTENSIONS.call_once(|| {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch("INSTALL ducklake; INSTALL sqlite;")
            .unwrap();
    });
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch("LOAD ducklake; LOAD sqlite;").unwrap();
    let data_path = catalog.with_extension("data");
    std::fs::create_dir_all(&data_path).unwrap();
    // No inlining, so the extension's own writes are data and delete files whose ids share
    // `next_file_id` with the crate's. Two deletes before the crate writes leave that
    // counter past the rowid SQLite would assign the crate's first file.
    conn.execute_batch(&format!(
        "ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0)",
        catalog.display(),
        data_path.display()
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

fn count_and_sum(conn: &duckdb::Connection, table: &str) -> (i64, i64) {
    conn.query_row(
        &format!("SELECT count(*)::BIGINT, coalesce(sum(id), 0)::BIGINT FROM lake.main.{table}"),
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .unwrap()
}

/// The ids of `columns`, each `(table, id column)`, as the catalog file holds them.
async fn ids(pool: &sqlx::SqlitePool, columns: &[(&str, &str)]) -> Vec<i64> {
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
async fn the_extension_reads_and_writes_a_catalog_after_the_crate_wrote_to_it() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.sqlite");
    {
        let lake = attach(&catalog);
        lake.execute_batch(
            "CREATE TABLE lake.main.orders (id BIGINT, amount DOUBLE);
             INSERT INTO lake.main.orders SELECT i, i FROM range(1, 11) r(i);
             DELETE FROM lake.main.orders WHERE id <= 2;
             DELETE FROM lake.main.orders WHERE id = 10;
             CREATE VIEW lake.main.big_orders AS SELECT * FROM lake.main.orders WHERE id > 5;",
        )
        .unwrap();
    }

    // The crate appends to the extension's table and creates a table of its own.
    let connection = format!("sqlite:{}?mode=rwc", catalog.display());
    let writer = SqliteMetadataWriter::new_with_init(&connection)
        .await
        .unwrap();
    let tables =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    tables
        .append_table("main", "orders", &[orders(&[11, 12, 13])])
        .await
        .unwrap();
    tables
        .write_table("main", "fresh", &[orders(&[100, 200])])
        .await
        .unwrap();
    drop(tables);

    {
        let lake = attach(&catalog);
        // 3..=9 the extension left, 11..=13 the crate appended.
        assert_eq!(count_and_sum(&lake, "orders"), (10, 78));
        assert_eq!(count_and_sum(&lake, "fresh"), (2, 300));
        // The extension allocates after the crate: a file, a delete file, a table.
        lake.execute_batch(
            "INSERT INTO lake.main.orders VALUES (14, 14);
             DELETE FROM lake.main.orders WHERE id = 11;
             CREATE TABLE lake.main.later AS SELECT 1::BIGINT AS id;",
        )
        .unwrap();
        assert_eq!(count_and_sum(&lake, "orders"), (10, 81));
        assert_eq!(count_and_sum(&lake, "later"), (1, 1));
        let views: i64 = lake
            .query_row("SELECT count(*) FROM lake.main.big_orders", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(views, 7);
    }

    // Neither writer handed out an id the other had taken, in either shared namespace.
    let pool = sqlx::SqlitePool::connect(&connection).await.unwrap();
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
    // Each file's statistics belong to a file the catalog holds: an id read back as the
    // SQLite rowid, which a catalog the extension created does not alias, would not be.
    let orphaned: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_file_column_stats
         WHERE data_file_id IS NULL
            OR data_file_id NOT IN (SELECT data_file_id FROM ducklake_data_file)",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        orphaned, 0,
        "column statistics name a file the catalog does not hold"
    );
    let unset: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_snapshot
         WHERE next_catalog_id IS NULL OR next_file_id IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(unset, 0, "a snapshot left the extension's allocators NULL");
}
