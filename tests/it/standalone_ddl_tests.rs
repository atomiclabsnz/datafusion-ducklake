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
use datafusion_ducklake::{
    DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
};
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

/// The statements both sides run, one snapshot each: DuckDB as SQL; this crate's
/// `CREATE SCHEMA` through DataFusion (`register_schema`), and its drops through the writer,
/// which is how a host's executor runs them (DataFusion's `deregister_*` would also be
/// reached by `CREATE OR REPLACE`).
enum Ddl {
    CreateSchema(&'static str),
    DropView(&'static str, &'static str),
    DropTable(&'static str, &'static str),
    DropSchema(&'static str),
}

const DDL: &[Ddl] = &[
    Ddl::DropView("s", "vw"),
    Ddl::DropTable("s", "t"),
    Ddl::CreateSchema("s2"),
    Ddl::DropSchema("s2"),
    Ddl::DropSchema("s"),
    Ddl::DropTable("main", "p"),
    Ddl::DropTable("main", "inl"),
];

impl Ddl {
    fn sql(&self) -> String {
        match self {
            Ddl::CreateSchema(s) => format!("CREATE SCHEMA lake.{s}"),
            Ddl::DropView(s, v) => format!("DROP VIEW lake.{s}.{v}"),
            Ddl::DropTable(s, t) => format!("DROP TABLE lake.{s}.{t}"),
            Ddl::DropSchema(s) => format!("DROP SCHEMA lake.{s}"),
        }
    }

    async fn run(&self, ctx: &SessionContext, writer: &dyn MetadataWriter) {
        let done = match self {
            Ddl::CreateSchema(_) => {
                ctx.sql(&self.sql()).await.unwrap().collect().await.unwrap();
                true
            },
            Ddl::DropView(s, v) => writer.drop_view(s, v).unwrap(),
            Ddl::DropTable(s, t) => writer.drop_table(s, t).unwrap(),
            Ddl::DropSchema(s) => writer.drop_schema(s).unwrap(),
        };
        assert!(done, "{}", self.sql());
    }
}

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

fn duckdb_with_inlining(catalog: &Path, data: &Path, limit: usize) -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite;
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT {limit});",
            catalog.display(),
            data.display()
        ))
        .unwrap();
    connection
}

async fn crate_context(catalog: &Path) -> (SessionContext, Arc<dyn MetadataWriter>) {
    let url = format!("sqlite:{}?mode=rw", catalog.display());
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(SqliteMetadataWriter::new(&url).await.unwrap());
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::clone(&writer)).unwrap()),
    );
    (ctx, writer)
}

/// Every row of every `ducklake_*` table, with a snapshot's time and a schema's or table's
/// UUID (which each side draws for itself) blanked.
///
/// Read through DuckDB's own sqlite scanner, so no second SQLite binding is needed here.
pub(crate) fn dump(catalog: &Path) -> Vec<String> {
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
    // `inl`'s rows inlined: attached with a limit rather than through `CALL set_option`, whose
    // bind crashes the bundled engine with the release extension (inside the extension's
    // `IsInlinedSystemColumn`); the CLI runs it fine.
    duckdb_with_inlining(&base, &data, 10)
        .execute_batch("INSERT INTO lake.main.inl VALUES (1), (2);")
        .unwrap();

    let official = temp.path().join("official.sqlite");
    let ours = temp.path().join("ours.sqlite");
    std::fs::copy(&base, &official).unwrap();
    std::fs::copy(&base, &ours).unwrap();

    let connection = duckdb(&official, &data);
    for statement in DDL {
        connection.execute_batch(&statement.sql()).unwrap();
    }
    drop(connection);

    let (ctx, writer) = crate_context(&ours).await;
    for statement in DDL {
        statement.run(&ctx, writer.as_ref()).await;
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
        .execute_batch(
            "CREATE SCHEMA lake.s; CREATE TABLE lake.s.x (i INT);
             CREATE SCHEMA lake.m; CREATE MACRO lake.m.f(x) AS x + 1;",
        )
        .unwrap();
    let (ctx, writer) = crate_context(&catalog).await;
    let snapshots = || {
        duckdb(&catalog, &data)
            .query_row("SELECT count(*) FROM lake.snapshots()", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
    };
    let before = snapshots();
    // A macro holds its schema too, as DuckDB counts it (`macro function "f" depends on
    // schema "m"`).
    let refused = writer
        .drop_schema("m")
        .expect_err("a schema holding a macro");
    assert!(
        matches!(refused, datafusion_ducklake::DuckLakeError::SchemaNotEmpty(ref m) if m.contains("\"f\"")),
        "{refused}"
    );
    let refused = writer
        .drop_schema("s")
        .expect_err("a schema holding a table");
    assert!(
        matches!(refused, datafusion_ducklake::DuckLakeError::SchemaNotEmpty(ref m) if m.contains("depends on schema")),
        "{refused}"
    );
    let refused = writer.create_schema("s", false).expect_err("a live schema");
    assert!(
        matches!(refused, datafusion_ducklake::DuckLakeError::SchemaExists(_)),
        "{refused}"
    );
    assert!(
        !writer.create_schema("s", true).unwrap(),
        "IF NOT EXISTS leaves it"
    );
    // DataFusion's CREATE SCHEMA asks the catalog first; DROP and CREATE OR REPLACE do not
    // reach the lake through DataFusion at all: main's refusal stands.
    let refused = ctx
        .sql("CREATE SCHEMA lake.s")
        .await
        .map(|_| ())
        .expect_err("already exists");
    assert!(refused.to_string().contains("already exists"), "{refused}");
    let refused = match ctx.sql("DROP TABLE lake.s.x").await {
        Ok(df) => df.collect().await.map(|_| ()),
        Err(e) => Err(e),
    }
    .expect_err("DataFusion does not drop a lake table");
    assert!(refused.to_string().contains("doesn't exist"), "{refused}");
    ctx.sql("CREATE SCHEMA IF NOT EXISTS lake.s")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(snapshots(), before, "nothing was written");
}

/// DataFusion's `CREATE OR REPLACE TABLE` deregisters the old table before it registers the
/// new one; the lake's schema has no `deregister_table`, so a replace there is refused before
/// anything is written, as before standalone DDL, and the table keeps its rows (#24 review).
#[tokio::test(flavor = "multi_thread")]
async fn create_or_replace_through_datafusion_drops_nothing() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("c.sqlite");
    duckdb(&catalog, &data)
        .execute_batch("CREATE TABLE lake.main.t AS SELECT 1 AS id;")
        .unwrap();
    let (ctx, _writer) = crate_context(&catalog).await;
    for sql in [
        "CREATE OR REPLACE TABLE lake.main.t AS SELECT 2 AS id",
        "CREATE OR REPLACE TABLE lake.main.t (id INT)",
    ] {
        let refused = match ctx.sql(sql).await {
            Ok(df) => df.collect().await.map(|_| ()),
            Err(e) => Err(e),
        };
        assert!(
            refused.is_err(),
            "{sql}: refused before anything is written"
        );
    }
    let rows: i64 = duckdb(&catalog, &data)
        .query_row("SELECT count(*) FROM lake.main.t WHERE id = 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(rows, 1, "the table and its row are as they were");
}

/// On a catalog this crate created itself (`initialize_schema`: no `schema_uuid`, ids numbered
/// per kind): `CREATE SCHEMA` and `DROP SCHEMA` work, and a drop ends no tag keyed by the same
/// number, which there can be another object's (#24 review).
#[tokio::test(flavor = "multi_thread")]
async fn ddl_on_a_catalog_this_crate_created() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("own.sqlite");
    let url = format!("sqlite:{}?mode=rwc", catalog.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "lake",
        Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), Arc::clone(&writer)).unwrap()),
    );
    let sql = |statement: &'static str| {
        let ctx = ctx.clone();
        async move { ctx.sql(statement).await.unwrap().collect().await.unwrap() }
    };
    sql("CREATE SCHEMA lake.x").await;

    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let schema_id: i64 = sqlx::query_scalar(
        "SELECT schema_id FROM ducklake_schema WHERE schema_name = 'x' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    // A tag keyed by the schema's number, as another object's comment would be.
    sqlx::query(
        "INSERT INTO ducklake_tag (object_id, begin_snapshot, end_snapshot, key, value)
         VALUES (?, 1, NULL, 'comment', 'another object')",
    )
    .bind(schema_id)
    .execute(&pool)
    .await
    .unwrap();
    let live_tags = || {
        let pool = pool.clone();
        async move {
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM ducklake_tag WHERE object_id = ? AND end_snapshot IS NULL",
            )
            .bind(schema_id)
            .fetch_one(&pool)
            .await
            .unwrap()
        }
    };
    assert_eq!(live_tags().await, 1);
    assert!(
        writer.drop_schema("x").unwrap(),
        "the empty schema is dropped"
    );
    assert_eq!(
        live_tags().await,
        1,
        "a tag keyed by the same number is not the schema's"
    );
}

/// A drop of one kind meets the other kind under the name: refused, as DuckDB refuses it
/// ("Existing object … is of type Table, trying to drop type View"), not "no such object"
/// (#24 review). And `CREATE SCHEMA` beside a schema a write's start has made, not committed,
/// takes that row up rather than making a second live schema of the name.
#[tokio::test(flavor = "multi_thread")]
async fn a_drop_of_the_wrong_kind_is_refused_and_a_pending_schema_is_taken_up() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("c.sqlite");
    duckdb(&catalog, &data)
        .execute_batch(
            "CREATE SCHEMA lake.s; CREATE TABLE lake.s.t (i INT);
             CREATE VIEW lake.s.v AS SELECT 1 AS one;",
        )
        .unwrap();
    let (_ctx, writer) = crate_context(&catalog).await;
    for (refused, what) in [
        (
            writer.drop_view("s", "t").expect_err("a table"),
            "DROP VIEW of a table",
        ),
        (
            writer.drop_table("s", "v").expect_err("a view"),
            "DROP TABLE of a view",
        ),
    ] {
        assert!(
            matches!(
                refused,
                datafusion_ducklake::DuckLakeError::WrongObjectKind(_)
            ),
            "{what}: {refused}"
        );
    }

    // A write's start has made the schema row of a schema not there yet: pending
    // (`begin_snapshot` = `PENDING_SNAPSHOT`, i64::MAX) until its commit stamps it.
    let url = format!("sqlite:{}", catalog.display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(
        "INSERT INTO ducklake_schema
             (schema_id, schema_uuid, begin_snapshot, end_snapshot, schema_name, path,
              path_is_relative)
         VALUES (900, NULL, 9223372036854775807, NULL, 'staging', 'staging/', 1)",
    )
    .execute(&pool)
    .await
    .unwrap();
    assert!(
        writer.create_schema("staging", false).unwrap(),
        "made, from the pending row"
    );
    let live: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_schema WHERE schema_name = 'staging' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live, 1, "one live schema of the name");
}
