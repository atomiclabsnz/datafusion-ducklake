//! Standalone DDL as its own snapshot (#18): `CREATE SCHEMA`, `DROP TABLE`, `DROP VIEW` and
//! `DROP SCHEMA` through DataFusion, on the SQLite catalog, against official DuckLake.
//!
//! One lake is built by DuckDB. A copy of its catalog takes the statements from DuckDB and
//! another from this crate, and the two catalogs must then hold the same rows, every table,
//! but for a snapshot's time and a schema's UUID. DuckDB then reads the crate's: what was
//! dropped is gone at the head and still there before it.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_ducklake::{DuckLakeCatalog, SqliteMetadataProvider, SqliteMetadataWriter};
use tempfile::TempDir;

/// The lake both sides start from, as DuckDB writes it: a schema holding a table with files,
/// a delete, a comment and a sort order, and a view over it; a partitioned table with a
/// column comment and an inlined one in `main`.
const SETUP: &str = "
    CREATE SCHEMA lake.s;
    CREATE TABLE lake.s.t (id INT, v VARCHAR);
    INSERT INTO lake.s.t SELECT i, 'x' || i FROM range(30) r(i);
    DELETE FROM lake.s.t WHERE id = 1;
    COMMENT ON TABLE lake.s.t IS 'c';
    ALTER TABLE lake.s.t SET SORTED BY (id);
    CREATE VIEW lake.s.vw AS SELECT * FROM lake.s.t;
    CREATE TABLE lake.main.p (id INT, k INT);
    ALTER TABLE lake.main.p SET PARTITIONED BY (k);
    INSERT INTO lake.main.p SELECT i, i % 2 FROM range(20) r(i);
    COMMENT ON COLUMN lake.main.p.id IS 'cc';
    CREATE TABLE lake.main.inl (id INT);
";

/// The statements both sides run, one snapshot each.
const DDL: &[&str] = &[
    "DROP VIEW lake.s.vw",
    "DROP TABLE lake.s.t",
    "CREATE SCHEMA lake.s2",
    "DROP SCHEMA lake.s2",
    "DROP SCHEMA lake.s",
    "DROP TABLE lake.main.p",
    "DROP TABLE lake.main.inl",
];

fn duckdb(catalog: &Path, data: &Path) -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite;
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0);",
            catalog.display(),
            data.display()
        ))
        .unwrap();
    connection
}

async fn crate_context(catalog: &Path) -> SessionContext {
    let url = format!("sqlite:{}?mode=rw", catalog.display());
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let writer = SqliteMetadataWriter::new(&url).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap()),
    );
    ctx
}

/// Every row of every `ducklake_*` table, with a snapshot's time and a schema's or table's
/// UUID (which each side draws for itself) blanked.
///
/// Read through DuckDB's own sqlite scanner, so no second SQLite binding is needed here.
fn dump(catalog: &Path) -> Vec<String> {
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD sqlite; ATTACH '{}' AS c (TYPE sqlite, READ_ONLY);",
            catalog.display()
        ))
        .unwrap();
    let tables: Vec<String> = connection
        .prepare(
            "SELECT table_name FROM information_schema.tables
             WHERE table_catalog = 'c' AND table_name LIKE 'ducklake%' ORDER BY table_name",
        )
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let uuid =
        regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}").unwrap();
    let mut rows = Vec::new();
    for table in tables {
        let blank_time = table == "ducklake_snapshot";
        let sql = format!(
            "SELECT * {} FROM c.\"{table}\" ORDER BY ALL",
            if blank_time {
                "EXCLUDE (snapshot_time)"
            } else {
                ""
            }
        );
        let mut statement = connection.prepare(&sql).unwrap();
        let mut result = statement.query([]).unwrap();
        while let Some(row) = result.next().unwrap() {
            // Known once the statement has run, so asked of the row.
            let columns = row.as_ref().column_count();
            let cells: Vec<String> = (0..columns)
                .map(|i| match row.get_ref(i).unwrap() {
                    duckdb::types::ValueRef::Null => "NULL".to_string(),
                    duckdb::types::ValueRef::Text(text) => {
                        String::from_utf8_lossy(text).into_owned()
                    },
                    value => format!("{value:?}"),
                })
                .collect();
            rows.push(
                uuid.replace_all(&format!("{table} {cells:?}"), "<uuid>")
                    .into_owned(),
            );
        }
    }
    rows
}

#[tokio::test(flavor = "multi_thread")]
async fn standalone_ddl_writes_the_catalog_duckdb_writes() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let base = temp.path().join("base.sqlite");
    duckdb(&base, &data).execute_batch(SETUP).unwrap();

    let official = temp.path().join("official.sqlite");
    let ours = temp.path().join("ours.sqlite");
    std::fs::copy(&base, &official).unwrap();
    std::fs::copy(&base, &ours).unwrap();

    let connection = duckdb(&official, &data);
    for statement in DDL {
        connection.execute_batch(statement).unwrap();
    }
    drop(connection);

    let ctx = crate_context(&ours).await;
    for statement in DDL {
        ctx.sql(statement)
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }

    let (official_rows, our_rows) = (dump(&official), dump(&ours));
    let missing: Vec<_> = official_rows
        .iter()
        .filter(|r| !our_rows.contains(r))
        .collect();
    let extra: Vec<_> = our_rows
        .iter()
        .filter(|r| !official_rows.contains(r))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "official's rows this crate did not write: {missing:#?}\nrows only this crate wrote: {extra:#?}"
    );

    // DuckDB reads the crate's lake: the dropped table is gone at the head and there before.
    let connection = duckdb(&ours, &data);
    // `main.p` is dropped by the last statement but one, so two snapshots before the head.
    let head: i64 = connection
        .query_row("SELECT max(snapshot_id) FROM lake.snapshots()", [], |row| {
            row.get(0)
        })
        .unwrap();
    let before: i64 = connection
        .query_row(
            &format!(
                "SELECT count(*) FROM lake.main.p AT (VERSION => {})",
                head - 2
            ),
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        before, 20,
        "time travel before the drop still reads the table"
    );
    assert!(
        connection
            .execute_batch("SELECT * FROM lake.main.p")
            .is_err(),
        "the table is gone at the head"
    );
}

/// `DROP SCHEMA` of a schema that still holds a table is refused, as DuckDB refuses it without
/// `CASCADE`, and writes nothing; `CASCADE` is refused by name; `CREATE SCHEMA` of a live
/// schema is refused, and `IF NOT EXISTS` leaves it.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_holding_a_table_is_not_dropped() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("c.sqlite");
    duckdb(&catalog, &data)
        .execute_batch("CREATE SCHEMA lake.s; CREATE TABLE lake.s.x (i INT);")
        .unwrap();
    let ctx = crate_context(&catalog).await;
    let snapshots = || {
        duckdb(&catalog, &data)
            .query_row("SELECT count(*) FROM lake.snapshots()", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
    };
    let before = snapshots();
    for (sql, expected) in [
        ("DROP SCHEMA lake.s", "depends on schema"),
        ("DROP SCHEMA lake.s CASCADE", "CASCADE"),
        ("CREATE SCHEMA lake.s", "already exists"),
    ] {
        let refused = match ctx.sql(sql).await {
            Ok(df) => df.collect().await.map(|_| ()),
            Err(e) => Err(e),
        }
        .expect_err(sql);
        assert!(refused.to_string().contains(expected), "{sql}: {refused}");
    }
    ctx.sql("CREATE SCHEMA IF NOT EXISTS lake.s")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(snapshots(), before, "nothing was written");
}
