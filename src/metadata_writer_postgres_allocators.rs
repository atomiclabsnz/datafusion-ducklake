//! Writing a Postgres catalog the DuckDB extension created.
//!
//! The extension's Postgres schema differs from this crate's in what a writer leans on:
//!
//! - its id columns are plain `BIGINT`s, not IDENTITY, so an insert that leaves the id
//!   to a sequence stores nothing (a NOT NULL refusal) — the id must be explicit;
//! - it allocates those ids from two watermarks on every `ducklake_snapshot` row,
//!   `next_catalog_id` (schemas, tables, views, partitions, sorts, macros) and
//!   `next_file_id` (data and delete files), and reads both at the head before any
//!   access, so a snapshot that leaves them NULL makes the catalog unreadable to it;
//! - it declares no key on `ducklake_table_stats`, so an `ON CONFLICT` seed has nothing
//!   to conflict on;
//! - it prunes a scan, inlined rows included, by the table-wide bounds in
//!   `ducklake_table_column_stats`, so bounds rebuilt from data files alone hide its
//!   inlined rows.
//!
//! Everything here is keyed on [`has_allocator_watermarks`]: this crate's own catalogs
//! carry neither watermark column, and for them every function is a no-op or answers
//! `None`, leaving their IDENTITY columns and counters as they were. The SQLite writer
//! has the same rules (`metadata_writer_sqlite`).

use crate::Result;
use sqlx::AssertSqlSafe;
use sqlx::PgConnection;
use sqlx::Row;

/// The ids the extension allocates from `next_catalog_id`.
const CATALOG_ID_COLUMNS: [(&str, &str); 6] = [
    ("ducklake_schema", "schema_id"),
    ("ducklake_table", "table_id"),
    ("ducklake_view", "view_id"),
    ("ducklake_partition_info", "partition_id"),
    ("ducklake_sort_info", "sort_id"),
    ("ducklake_macro", "macro_id"),
];

/// The ids the extension allocates from `next_file_id`.
const FILE_ID_COLUMNS: [(&str, &str); 2] =
    [("ducklake_data_file", "data_file_id"), ("ducklake_delete_file", "delete_file_id")];

/// Whether this catalog carries the extension's id allocators (see the module doc).
pub(crate) async fn has_allocator_watermarks(conn: &mut PgConnection) -> Result<bool> {
    let columns: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pg_attribute
         WHERE attrelid = to_regclass('ducklake_snapshot')
           AND attname IN ('next_catalog_id', 'next_file_id') AND NOT attisdropped",
    )
    .fetch_one(&mut *conn)
    .await?;
    Ok(columns == 2)
}

/// The first id free under `watermark`: past the largest watermark any snapshot records
/// and past every id in `columns`, of the tables this catalog has. The extension can
/// advance a watermark past the ids it leaves in the tables (an inlined insert advances
/// `next_file_id`, and a cleanup removes file rows), so neither bound alone is enough.
async fn first_free_id(
    conn: &mut PgConnection,
    watermark: &str,
    columns: &[(&str, &str)],
) -> Result<i64> {
    let mut bounds = vec![format!("SELECT MAX({watermark}) AS id FROM ducklake_snapshot")];
    for (table, column) in columns {
        let exists: bool = sqlx::query_scalar("SELECT to_regclass($1) IS NOT NULL")
            .bind(*table)
            .fetch_one(&mut *conn)
            .await?;
        if exists {
            bounds.push(format!("SELECT MAX({column}) + 1 FROM {table}"));
        }
    }
    // Every name in the statement is one of the constant table and column names above.
    let id: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT COALESCE(MAX(id), 0)::BIGINT FROM ({}) AS bounds",
        bounds.join(" UNION ALL ")
    )))
    .fetch_one(&mut *conn)
    .await?;
    Ok(id)
}

/// The id for a new data or delete file: `None` in this crate's own catalogs, where the
/// insert's `COALESCE($id, nextval(…))` takes the IDENTITY sequence as before.
pub(crate) async fn next_file_id(conn: &mut PgConnection) -> Result<Option<i64>> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(None);
    }
    Ok(Some(
        first_free_id(conn, "next_file_id", &FILE_ID_COLUMNS).await?,
    ))
}

/// The id for a new schema, table, partition or sort: `None` in this crate's own
/// catalogs, which keep allocating from their sequences and counters.
pub(crate) async fn next_catalog_id(conn: &mut PgConnection) -> Result<Option<i64>> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(None);
    }
    Ok(Some(
        first_free_id(conn, "next_catalog_id", &CATALOG_ID_COLUMNS).await?,
    ))
}

/// The snapshot id to commit: `counted`, the crate's own counter's next value, or — in a
/// catalog the extension also writes, whose snapshots that counter never saw — one past
/// the newest snapshot, if that is later.
pub(crate) async fn next_snapshot_id(conn: &mut PgConnection, counted: i64) -> Result<i64> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(counted);
    }
    let newest: Option<i64> = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&mut *conn)
        .await?;
    Ok(counted.max(newest.map_or(0, |id| id + 1)))
}

/// Leave the head snapshot's watermarks past every id in the catalog. Every commit of a
/// writer that may meet such a catalog calls this just before `COMMIT`, so no snapshot it
/// creates is left with NULL watermarks. The head is written only when its watermarks are
/// behind, so a transaction that only read stays a read.
pub(crate) async fn stamp_allocator_watermarks(conn: &mut PgConnection) -> Result<()> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(());
    }
    let catalog = first_free_id(conn, "next_catalog_id", &CATALOG_ID_COLUMNS).await?;
    let file = first_free_id(conn, "next_file_id", &FILE_ID_COLUMNS).await?;
    let head = sqlx::query(
        "SELECT snapshot_id, next_catalog_id, next_file_id FROM ducklake_snapshot
         ORDER BY snapshot_id DESC LIMIT 1",
    )
    .fetch_optional(&mut *conn)
    .await?;
    if let Some(head) = head {
        let snapshot_id: i64 = head.try_get(0)?;
        let head_catalog: Option<i64> = head.try_get(1)?;
        let head_file: Option<i64> = head.try_get(2)?;
        if head_catalog != Some(catalog) || head_file != Some(file) {
            sqlx::query(
                "UPDATE ducklake_snapshot SET next_catalog_id = $1, next_file_id = $2
                 WHERE snapshot_id = $3",
            )
            .bind(catalog)
            .bind(file)
            .bind(snapshot_id)
            .execute(&mut *conn)
            .await?;
        }
    }
    Ok(())
}

/// Whether the table-wide bounds must be recorded as unknown rather than rolled up from
/// data files: in a catalog the extension created, while any of the table's inlined rows
/// is live (see the module doc). Always `false` in this crate's own catalogs.
pub(crate) async fn bounds_must_be_unknown(conn: &mut PgConnection, table_id: i64) -> Result<bool> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(false);
    }
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT i.table_name FROM ducklake_inlined_data_tables i
         WHERE i.table_id = $1 AND to_regclass(quote_ident(i.table_name)) IS NOT NULL",
    )
    .bind(table_id)
    .fetch_all(&mut *conn)
    .await?;
    for table in tables {
        let live: bool = sqlx::query_scalar(AssertSqlSafe(format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE end_snapshot IS NULL)",
            crate::metadata_writer_postgres::quote_ident(&table)
        )))
        .fetch_one(&mut *conn)
        .await?;
        if live {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Record every live top-level column's table-wide bounds as unknown (NULL), which no
/// reader prunes by and the extension keeps unknown when it writes next.
pub(crate) async fn forget_column_bounds(conn: &mut PgConnection, table_id: i64) -> Result<()> {
    let column_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT column_id FROM ducklake_column
         WHERE table_id = $1 AND end_snapshot IS NULL AND parent_column IS NULL",
    )
    .bind(table_id)
    .fetch_all(&mut *conn)
    .await?;
    sqlx::query("DELETE FROM ducklake_table_column_stats WHERE table_id = $1")
        .bind(table_id)
        .execute(&mut *conn)
        .await?;
    for column_id in column_ids {
        sqlx::query(
            "INSERT INTO ducklake_table_column_stats
                 (table_id, column_id, contains_null, contains_nan, min_value, max_value, extra_stats)
             VALUES ($1, $2, NULL, NULL, NULL, NULL, NULL)",
        )
        .bind(table_id)
        .bind(column_id)
        .execute(&mut *conn)
        .await?;
    }
    Ok(())
}

/// Seed the table's `ducklake_table_stats` row if it has none. This crate's own catalogs
/// key the table on `table_id`, and `ON CONFLICT` keeps two writers seeding at once from
/// colliding; the extension's schema declares no key, so there the seed is guarded by
/// `NOT EXISTS` — its writers serialize on the snapshot id, so they cannot race here.
pub(crate) async fn seed_table_stats(conn: &mut PgConnection, table_id: i64) -> Result<()> {
    let sql = if has_allocator_watermarks(conn).await? {
        "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
         SELECT $1, 0, 0, 0
         WHERE NOT EXISTS (SELECT 1 FROM ducklake_table_stats WHERE table_id = $1)"
    } else {
        "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
         VALUES ($1, 0, 0, 0)
         ON CONFLICT DO NOTHING"
    };
    sqlx::query(sql).bind(table_id).execute(&mut *conn).await?;
    Ok(())
}
