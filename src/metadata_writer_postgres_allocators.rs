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
//!   to conflict on.
//!
//! Everything here is keyed on [`has_allocator_watermarks`]: this crate's own catalogs
//! carry neither watermark column, and for them every function is a no-op or answers
//! `None`, leaving their IDENTITY columns and counters as they were. The SQLite writer
//! has the same rules (`metadata_writer_sqlite`).

use crate::Result;
use sqlx::AssertSqlSafe;
use sqlx::PgConnection;

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

/// Take `SHARE ROW EXCLUSIVE` on every table whose ids the extension's watermarks
/// allocate, for the rest of the transaction, before reading a free id. The mode conflicts
/// with the `ROW EXCLUSIVE` every `INSERT` takes, so a concurrent writer's uncommitted row
/// — the extension's or another of this crate's — is waited for and then seen, and none
/// can insert until this transaction ends: no two writers hand out one id, though the
/// extension's tables declare no key that would stop them (`ducklake_table` has none).
/// Always every such table, in one fixed order, so two allocating writers cannot lock in
/// different orders.
async fn lock_allocated_tables(conn: &mut PgConnection) -> Result<()> {
    let mut tables: Vec<&str> = CATALOG_ID_COLUMNS
        .iter()
        .chain(FILE_ID_COLUMNS.iter())
        .map(|(table, _)| *table)
        .collect();
    tables.sort_unstable();
    let mut present = Vec::with_capacity(tables.len());
    for table in tables {
        let exists: bool = sqlx::query_scalar("SELECT to_regclass($1) IS NOT NULL")
            .bind(table)
            .fetch_one(&mut *conn)
            .await?;
        if exists {
            present.push(table);
        }
    }
    // Every name in the statement is one of the constant table names above.
    sqlx::query(AssertSqlSafe(format!(
        "LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE",
        present.join(", ")
    )))
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// A stand-in for a schema or table id a write reserves before it commits, in a catalog
/// the extension created: there is no sequence to reserve from, and the real id is
/// allocated at commit, under [`lock_allocated_tables`]. Negative and distinct per call,
/// so no stand-in is mistaken for another, or for an id the catalog holds. `None` in this
/// crate's own catalogs, which reserve from their IDENTITY sequences.
pub(crate) async fn placeholder_id(conn: &mut PgConnection) -> Result<Option<i64>> {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);
    if !has_allocator_watermarks(conn).await? {
        return Ok(None);
    }
    Ok(Some(
        NEXT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed),
    ))
}

/// The id for a new data or delete file: `None` in this crate's own catalogs, where the
/// insert's `COALESCE($id, nextval(…))` takes the IDENTITY sequence as before.
pub(crate) async fn next_file_id(conn: &mut PgConnection) -> Result<Option<i64>> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(None);
    }
    lock_allocated_tables(conn).await?;
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
    lock_allocated_tables(conn).await?;
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
/// behind, and only ever raised.
pub(crate) async fn stamp_allocator_watermarks(conn: &mut PgConnection) -> Result<()> {
    if !has_allocator_watermarks(conn).await? {
        return Ok(());
    }
    let catalog = first_free_id(conn, "next_catalog_id", &CATALOG_ID_COLUMNS).await?;
    let file = first_free_id(conn, "next_file_id", &FILE_ID_COLUMNS).await?;
    // Only ever raised: a head another writer committed after the ids above were read
    // keeps its own watermarks where they are higher.
    sqlx::query(
        "UPDATE ducklake_snapshot
         SET next_catalog_id = GREATEST(COALESCE(next_catalog_id, 0), $1),
             next_file_id = GREATEST(COALESCE(next_file_id, 0), $2)
         WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM ducklake_snapshot)
           AND (next_catalog_id IS NULL OR next_catalog_id < $1
                OR next_file_id IS NULL OR next_file_id < $2)",
    )
    .bind(catalog)
    .bind(file)
    .execute(&mut *conn)
    .await?;
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

/// Raise the crate's `next_column_id` counter past every column id the catalog holds,
/// before reserving from it. The extension numbers a table's columns from that table's own
/// highest id, so a counter seeded once falls behind a column it added since, and a column
/// this crate added next would take that column's id. Raised in every catalog — a counter
/// already ahead is left alone.
pub(crate) async fn raise_column_counter(conn: &mut PgConnection) -> Result<()> {
    sqlx::query(
        "UPDATE ducklake_metadata
         SET value = CAST(GREATEST(CAST(value AS BIGINT),
                     (SELECT COALESCE(MAX(column_id), 0) FROM ducklake_column)) AS VARCHAR)
         WHERE key = 'next_column_id' AND scope IS NULL",
    )
    .execute(&mut *conn)
    .await?;
    Ok(())
}
