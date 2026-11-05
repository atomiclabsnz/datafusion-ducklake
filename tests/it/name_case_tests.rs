//! Catalog names and case, against official DuckLake (ferrion#708,
//! atomiclabsnz/datafusion-ducklake#17 item 1).
//!
//! The rule, as DuckDB 1.5.6 keeps it over a SQLite catalog (pinned by
//! [`duckdb_compares_names_ascii_case_insensitively_and_stores_them_as_written`]):
//!
//! - A schema, table or view name is **stored as written**, quoted or not: `"Staging"`,
//!   `"Orders"`, `MixedUnquoted`. A `CREATE OR REPLACE` stores the new spelling.
//! - It is **found in any ASCII case**: `lake.main.orders`, `lake.MAIN.ORDERS` and
//!   `lake.main."ORDERS"` all name `"Orders"`, for a read, an insert, a drop, `IF EXISTS` and
//!   `IF NOT EXISTS`.
//! - So two live names in one namespace may not differ only by ASCII case: `CREATE SCHEMA
//!   lake.staging` beside `"Staging"`, a table `orders` beside `"Orders"`, a table `sales` beside
//!   a view `"Sales"` and a view `orders` beside a table `"Orders"` are all refused.
//! - **Only ASCII** folds: `"Ärger"` and `"ärger"` are two tables, and neither finds the other.
//!
//! The crate follows it on every catalog backend, with one comparison per SQL dialect
//! (`catalog_name::name_eq!`): this file checks it on the SQLite catalog, a lake DuckDB wrote read
//! and written here and then read back by DuckDB, and a lake written here read by DuckDB.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

/// DuckDB over the SQLite catalog, rows in Parquet.
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

/// What DuckDB answers for `sql`, one line per row, or its error's message.
fn duckdb_answer(connection: &duckdb::Connection, sql: &str) -> Result<Vec<String>, String> {
    let mut statement = connection.prepare(sql).map_err(|e| e.to_string())?;
    let mut result = statement.query([]).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    while let Some(row) = result.next().map_err(|e| e.to_string())? {
        let columns = row.as_ref().column_count();
        let cells: Vec<String> = (0..columns)
            .map(|i| match row.get_ref(i).unwrap() {
                duckdb::types::ValueRef::Text(text) => String::from_utf8_lossy(text).into_owned(),
                duckdb::types::ValueRef::Int(value) => value.to_string(),
                duckdb::types::ValueRef::BigInt(value) => value.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        rows.push(cells.join("|"));
    }
    Ok(rows)
}

fn rows(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    duckdb_answer(connection, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The refusal DuckDB answers `sql` with; it must start with `expected`.
fn refused(connection: &duckdb::Connection, sql: &str, expected: &str) {
    match duckdb_answer(connection, sql) {
        Ok(rows) => panic!("DuckDB ran {sql}: {rows:?}"),
        Err(message) => assert!(
            message.starts_with(expected),
            "{sql}: DuckDB refused with {message:?}, not {expected:?}"
        ),
    }
}

/// The live names of each kind in the catalog, as stored.
fn live_names(connection: &duckdb::Connection) -> Vec<String> {
    rows(
        connection,
        "SELECT 'schema ' || schema_name FROM __ducklake_metadata_lake.ducklake_schema
          WHERE end_snapshot IS NULL
         UNION ALL
         SELECT 'table ' || table_name FROM __ducklake_metadata_lake.ducklake_table
          WHERE end_snapshot IS NULL
         UNION ALL
         SELECT 'view ' || view_name FROM __ducklake_metadata_lake.ducklake_view
          WHERE end_snapshot IS NULL
         ORDER BY 1",
    )
}

fn new_lake(temp: &TempDir) -> (PathBuf, PathBuf) {
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    (temp.path().join("catalog.sqlite"), data)
}

/// The oracle. Every statement here and its outcome (DuckDB's message verbatim, up to its
/// "Did you mean" hint) is what DuckDB 1.5.6 does; the crate's tests below hold it to this.
#[test]
fn duckdb_compares_names_ascii_case_insensitively_and_stores_them_as_written() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    let lake = duckdb(&catalog, &data);
    lake.execute_batch(
        r#"CREATE SCHEMA lake."Staging";
           CREATE TABLE lake.main."Orders" (id INT);
           CREATE TABLE lake.main.MixedUnquoted (id INT);
           CREATE VIEW lake.main."Sales" AS SELECT 1 AS x;"#,
    )
    .unwrap();
    // Stored as written, quoted or not.
    assert_eq!(
        live_names(&lake),
        ["schema Staging", "schema main", "table MixedUnquoted", "table Orders", "view Sales"]
    );

    // Found in any case, by a write and a read.
    rows(&lake, "INSERT INTO lake.main.orders VALUES (1)");
    rows(&lake, "INSERT INTO lake.MAIN.ORDERS VALUES (2)");
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.main.orders ORDER BY id"),
        ["1", "2"]
    );
    assert_eq!(
        rows(&lake, r#"SELECT count(*) FROM lake.Main."ORDERS""#),
        ["2"]
    );
    assert_eq!(rows(&lake, "SELECT x FROM lake.main.SALES"), ["1"]);
    rows(&lake, "CREATE TABLE lake.STAGING.t (id INT)");

    // Two live names of one namespace may not differ only by case, whatever their kinds.
    refused(
        &lake,
        "CREATE SCHEMA lake.staging",
        r#"Catalog Error: Schema with name "staging" already exists!"#,
    );
    rows(&lake, "CREATE SCHEMA IF NOT EXISTS lake.STAGING");
    refused(
        &lake,
        "CREATE TABLE lake.main.orders (id INT)",
        r#"Catalog Error: Table with name "orders" already exists!"#,
    );
    rows(
        &lake,
        "CREATE TABLE IF NOT EXISTS lake.main.ORDERS (id INT)",
    );
    refused(
        &lake,
        "CREATE TABLE lake.main.sales (id INT)",
        r#"Catalog Error: View with name "sales" already exists!"#,
    );
    refused(
        &lake,
        "CREATE VIEW lake.main.orders AS SELECT 1 AS x",
        r#"Catalog Error: Table with name "orders" already exists!"#,
    );
    assert_eq!(
        live_names(&lake),
        [
            "schema Staging",
            "schema main",
            "table MixedUnquoted",
            "table Orders",
            "table t",
            "view Sales"
        ],
        "a refused statement and an IF NOT EXISTS added nothing"
    );

    // A replace stores the new spelling.
    rows(
        &lake,
        "CREATE OR REPLACE TABLE lake.main.ORDERS (id INT, b INT)",
    );
    rows(
        &lake,
        "CREATE OR REPLACE VIEW lake.main.sales AS SELECT 2 AS x",
    );
    assert_eq!(
        live_names(&lake),
        [
            "schema Staging",
            "schema main",
            "table MixedUnquoted",
            "table ORDERS",
            "table t",
            "view sales"
        ]
    );

    // A drop finds the name in any case; the wrong kind is refused even with IF EXISTS.
    refused(
        &lake,
        "DROP VIEW lake.main.mixedunquoted",
        "Catalog Error: Existing object MixedUnquoted is of type Table, trying to drop type View",
    );
    refused(
        &lake,
        "DROP VIEW IF EXISTS lake.main.mixedunquoted",
        "Catalog Error: Existing object MixedUnquoted is of type Table, trying to drop type View",
    );
    refused(
        &lake,
        "DROP SCHEMA lake.staging",
        r#"Catalog Error: Cannot drop schema "Staging" because there are entries that depend on it"#,
    );
    rows(&lake, "DROP TABLE lake.main.orders");
    rows(&lake, "DROP TABLE IF EXISTS lake.main.Orders");
    refused(
        &lake,
        "DROP TABLE lake.main.orders",
        "Catalog Error: Table with name orders does not exist!",
    );
    rows(&lake, "DROP VIEW lake.main.SALES");
    rows(&lake, "DROP TABLE lake.Staging.T");
    rows(&lake, "DROP SCHEMA lake.STAGING");
    assert_eq!(live_names(&lake), ["schema main", "table MixedUnquoted"]);

    // A rename to another case of its own name is a rename; to another's, refused.
    rows(
        &lake,
        "ALTER TABLE lake.main.mixedunquoted RENAME TO mixedunquoted",
    );
    rows(&lake, "CREATE TABLE lake.main.other (id INT)");
    refused(
        &lake,
        r#"ALTER TABLE lake.main.other RENAME TO "MIXEDUNQUOTED""#,
        "Binder Error: Cannot rename table other to MIXEDUNQUOTED, since MIXEDUNQUOTED already \
         exists.",
    );

    // Only ASCII folds.
    rows(&lake, r#"CREATE TABLE lake.main."Ärger" (id INT)"#);
    refused(
        &lake,
        r#"SELECT * FROM lake.main."ärger""#,
        "Catalog Error: Table with name ärger does not exist!",
    );
    rows(&lake, r#"CREATE TABLE lake.main."ärger" (id INT)"#);
    assert_eq!(
        live_names(&lake),
        ["schema main", "table mixedunquoted", "table other", "table Ärger", "table ärger"]
    );
}

/// DataFusion over the catalog, writable, and the writer behind it.
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

/// A table writer over the catalog, rows in Parquet. Opening it brings DuckDB's catalog up to
/// what this crate writes, once.
async fn table_writer(catalog: &Path) -> DuckLakeTableWriter {
    let url = format!("sqlite:{}?mode=rw", catalog.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer.with_data_inlining_row_limit(0));
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<String> = ids.iter().map(|id| format!("row {id}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

/// What DataFusion answers for `sql`: its rows, one line each, or its error's message.
async fn crate_answer(ctx: &SessionContext, sql: &str) -> Result<Vec<String>, String> {
    let batches = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .collect()
        .await
        .map_err(|e| e.to_string())?;
    let text = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    // The data lines of the table, without its borders and header.
    Ok(text
        .lines()
        .filter(|line| line.starts_with('|'))
        .skip(1)
        .map(|line| {
            line.trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect())
}

/// The lake DuckDB writes with mixed-case names: schema `"Staging"` holding table `"Orders"`
/// with two rows, view `"Sales"` and table `"Ärger"` in `main`.
fn mixed_case_lake(temp: &TempDir) -> (PathBuf, PathBuf) {
    let (catalog, data) = new_lake(temp);
    duckdb(&catalog, &data)
        .execute_batch(
            r#"CREATE SCHEMA lake."Staging";
               CREATE TABLE lake."Staging"."Orders" (id INT, name VARCHAR);
               INSERT INTO lake."Staging"."Orders" VALUES (1, 'one'), (2, 'two');
               CREATE VIEW lake.main."Sales" AS SELECT 7 AS id;
               CREATE TABLE lake.main."Ärger" (id INT, name VARCHAR);"#,
        )
        .unwrap();
    (catalog, data)
}

const MIXED_CASE_NAMES: [&str; 5] =
    ["schema Staging", "schema main", "table Orders", "table Ärger", "view Sales"];

#[tokio::test(flavor = "multi_thread")]
async fn a_lake_duckdb_wrote_in_mixed_case_is_read_and_written_here_in_any_case() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = mixed_case_lake(&temp);
    table_writer(&catalog).await;
    let (ctx, _) = crate_context(&catalog).await;

    for sql in [
        "SELECT count(*) FROM lake.staging.orders",
        "SELECT count(*) FROM lake.STAGING.Orders",
        r#"SELECT count(*) FROM lake."STAGING"."ORDERS""#,
        r#"SELECT count(*) FROM lake."Staging"."Orders""#,
    ] {
        assert_eq!(
            crate_answer(&ctx, sql).await,
            Ok(vec!["2".to_string()]),
            "{sql}"
        );
    }
    assert_eq!(
        crate_answer(&ctx, "SELECT id FROM lake.main.sales").await,
        Ok(vec!["7".to_string()])
    );
    // Only ASCII folds: `ärger` is not `Ärger`, for DuckDB or here.
    let error = crate_answer(&ctx, r#"SELECT * FROM lake.main."ärger""#)
        .await
        .unwrap_err();
    assert!(error.contains("not found"), "{error}");

    // An insert through an unquoted name lands in DuckDB's `"Orders"`.
    crate_answer(&ctx, "INSERT INTO lake.staging.orders VALUES (3, 'three')")
        .await
        .unwrap();
    let writer = table_writer(&catalog).await;
    writer
        .append_table("STAGING", "ORDERS", &[batch(&[4])])
        .await
        .unwrap();

    let lake = duckdb(&catalog, &data);
    assert_eq!(
        live_names(&lake),
        MIXED_CASE_NAMES,
        "no second schema or table"
    );
    assert_eq!(
        rows(
            &lake,
            r#"SELECT id FROM lake."Staging"."Orders" ORDER BY id"#
        ),
        ["1", "2", "3", "4"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_name_differing_only_by_case_is_refused_as_duckdb_refuses_it() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = mixed_case_lake(&temp);
    let writer = table_writer(&catalog).await;
    let (ctx, metadata) = crate_context(&catalog).await;
    let catalog_before = crate::standalone_ddl_tests::dump(&catalog);

    // A schema.
    match metadata.create_schema("staging", false) {
        Err(DuckLakeError::SchemaExists(_)) => {},
        other => panic!("CREATE SCHEMA staging beside \"Staging\": {other:?}"),
    }
    assert!(!metadata.create_schema("STAGING", true).unwrap());
    assert!(
        crate_answer(&ctx, "CREATE SCHEMA lake.staging")
            .await
            .is_err()
    );
    crate_answer(&ctx, "CREATE SCHEMA IF NOT EXISTS lake.staging")
        .await
        .unwrap();

    // A table beside a table, through SQL and the writer.
    assert!(
        crate_answer(&ctx, "CREATE TABLE lake.staging.orders (id INT)")
            .await
            .is_err()
    );
    crate_answer(
        &ctx,
        "CREATE TABLE IF NOT EXISTS lake.staging.orders (id INT)",
    )
    .await
    .unwrap();
    match writer
        .create_table("staging", "orders", &[batch(&[9])])
        .await
    {
        Err(DuckLakeError::TableExists(_)) => {},
        other => panic!("a table orders beside \"Orders\": {other:?}"),
    }

    // A table beside a view, in every mode; named as the catalog names the view.
    for mode in [WriteMode::Create, WriteMode::Replace, WriteMode::Append] {
        match writer.begin_write("main", "SALES", &batch(&[]).schema(), mode) {
            Err(DuckLakeError::ViewExists(message)) => {
                // Named as the write named it, as DuckDB's refusal does.
                assert!(message.starts_with("main.SALES:"), "{message}")
            },
            Err(other) => panic!("{mode:?}: a table SALES beside view \"Sales\": {other:?}"),
            Ok(_) => panic!("{mode:?}: a table SALES begun beside view \"Sales\""),
        }
    }

    assert_eq!(
        crate::standalone_ddl_tests::dump(&catalog),
        catalog_before,
        "a refused statement changed the catalog"
    );
    let lake = duckdb(&catalog, &data);
    assert_eq!(live_names(&lake), MIXED_CASE_NAMES);
    assert_eq!(
        rows(&lake, "SELECT count(*) FROM lake.staging.orders"),
        ["2"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drop_finds_the_name_in_any_case() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = mixed_case_lake(&temp);
    table_writer(&catalog).await;
    let (_, writer) = crate_context(&catalog).await;

    match writer.drop_view("staging", "orders") {
        Err(DuckLakeError::WrongObjectKind(_)) => {},
        other => panic!("DROP VIEW of table \"Orders\": {other:?}"),
    }
    match writer.drop_table("main", "SALES") {
        Err(DuckLakeError::WrongObjectKind(_)) => {},
        other => panic!("DROP TABLE of view \"Sales\": {other:?}"),
    }
    match writer.drop_schema("staging") {
        Err(DuckLakeError::SchemaNotEmpty(_)) => {},
        other => panic!("DROP SCHEMA of \"Staging\", holding \"Orders\": {other:?}"),
    }
    // dbt's `drop table if exists`, which is how `--full-refresh` drops what it replaces.
    assert!(writer.drop_table("STAGING", "orders").unwrap());
    assert!(
        !writer.drop_table("staging", "Orders").unwrap(),
        "dropped twice"
    );
    assert!(writer.drop_view("main", "sales").unwrap());
    assert!(writer.drop_schema("staging").unwrap());
    // Only ASCII folds.
    assert!(!writer.drop_table("main", "ärger").unwrap());

    let lake = duckdb(&catalog, &data);
    assert_eq!(live_names(&lake), ["schema main", "table Ärger"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn create_or_replace_stores_the_new_spelling_as_duckdb_does() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = mixed_case_lake(&temp);
    let writer = table_writer(&catalog).await;
    writer
        .create_or_replace_table("staging", "ORDERS", &[batch(&[5, 6, 7])])
        .await
        .unwrap();
    let lake = duckdb(&catalog, &data);
    assert_eq!(
        live_names(&lake),
        ["schema Staging", "schema main", "table ORDERS", "table Ärger", "view Sales"]
    );
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.staging.orders ORDER BY id"),
        ["5", "6", "7"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lake_written_here_in_mixed_case_is_found_by_duckdb_in_any_case() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    // DuckDB makes the catalog; everything in it is written here.
    duckdb(&catalog, &data);
    let writer = table_writer(&catalog).await;
    let (ctx, metadata) = crate_context(&catalog).await;
    assert!(metadata.create_schema("Fin", false).unwrap());
    writer
        .create_table("fin", "Ledger", &[batch(&[1, 2])])
        .await
        .unwrap();
    writer
        .append_table("FIN", "ledger", &[batch(&[3])])
        .await
        .unwrap();
    match writer.create_table("Fin", "LEDGER", &[batch(&[9])]).await {
        Err(DuckLakeError::TableExists(_)) => {},
        other => panic!("a table LEDGER beside \"Ledger\": {other:?}"),
    }
    // `"Ärger"` and `"ärger"` are two tables.
    writer
        .create_table("main", "Ärger", &[batch(&[1])])
        .await
        .unwrap();
    writer
        .create_table("main", "ärger", &[batch(&[2, 3])])
        .await
        .unwrap();
    assert_eq!(
        crate_answer(&ctx, "SELECT count(*) FROM lake.fin.ledger").await,
        Ok(vec!["3".to_string()])
    );

    let lake = duckdb(&catalog, &data);
    assert_eq!(
        live_names(&lake),
        ["schema Fin", "schema main", "table Ledger", "table Ärger", "table ärger"]
    );
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.fin.ledger ORDER BY id"),
        ["1", "2", "3"]
    );
    // Its creation is recorded as DuckDB records one: the schema as stored, the table as its
    // create spelled it.
    let created = rows(
        &lake,
        "SELECT changes_made FROM __ducklake_metadata_lake.ducklake_snapshot_changes
          WHERE changes_made LIKE '%created_table%' ORDER BY snapshot_id",
    );
    assert!(
        created[0].contains(r#"created_table:"Fin"."Ledger""#),
        "{created:?}"
    );
    assert_eq!(
        rows(&lake, r#"SELECT count(*) FROM lake.main."ärger""#),
        ["2"]
    );
    refused(
        &lake,
        "CREATE TABLE lake.FIN.LEDGER (id INT)",
        r#"Catalog Error: Table with name "LEDGER" already exists!"#,
    );
    // DuckDB goes on writing the table it finds.
    rows(&lake, "INSERT INTO lake.fin.LEDGER VALUES (4, 'four')");
    assert_eq!(rows(&lake, "SELECT count(*) FROM lake.Fin.Ledger"), ["4"]);
}

/// Every Parquet file under `dir`, relative to it.
fn parquet_files(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, found);
            } else if path.extension().is_some_and(|e| e == "parquet") {
                let relative = path.strip_prefix(root).unwrap().parent().unwrap();
                found.push(relative.display().to_string());
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, dir, &mut found);
    found.sort();
    found.dedup();
    found
}

/// Directory names as `dir`'s filesystem tells them apart. A case-insensitive one (macOS's
/// default APFS) holds `main/ORDERS` and `main/orders` as one directory, so there the names
/// are compared folded; a reader finds every file either way, by its row's path.
fn as_this_filesystem_names<S: AsRef<str>>(
    dir: &Path,
    names: impl IntoIterator<Item = S>,
) -> Vec<String> {
    let probe = dir.join("CaseProbe");
    std::fs::write(&probe, b"").unwrap();
    let folds = dir.join("caseprobe").exists();
    std::fs::remove_file(&probe).unwrap();
    let mut out: Vec<String> = names
        .into_iter()
        .map(|n| {
            let n = n.as_ref();
            if folds {
                n.to_ascii_lowercase()
            } else {
                n.to_string()
            }
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The directories DuckDB 1.5.6 gives a schema and table: its name as written, then kept by a
/// rename. Pinned here, as the writer's keys follow them.
#[test]
fn duckdb_keeps_a_renamed_tables_path() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    let lake = duckdb(&catalog, &data);
    lake.execute_batch(
        r#"CREATE SCHEMA lake."Staging";
           CREATE TABLE lake."Staging"."Orders" (id INT);
           CREATE TABLE lake.main.orders (id INT);
           INSERT INTO lake.main.orders VALUES (1);
           ALTER TABLE lake.main.orders RENAME TO "Orders";
           INSERT INTO lake.main.orders VALUES (2);
           CREATE TABLE lake.main.a (id INT);
           ALTER TABLE lake.main.a RENAME TO b;
           INSERT INTO lake.main.b VALUES (3);"#,
    )
    .unwrap();
    assert_eq!(
        rows(
            &lake,
            "SELECT s.schema_name || '|' || s.path || '|' || t.table_name || '|' || t.path
               FROM __ducklake_metadata_lake.ducklake_table t
               JOIN __ducklake_metadata_lake.ducklake_schema s USING (schema_id)
              WHERE t.end_snapshot IS NULL ORDER BY t.table_id"
        ),
        ["Staging|Staging/|Orders|Orders/", "main|main/|Orders|orders/", "main|main/|b|a/"]
    );
    // DuckDB's own inserts after the renames go to the kept paths.
    assert_eq!(parquet_files(&data), ["main/a", "main/orders"]);
}

/// A table DuckDB renamed keeps its `path`; a write here, by any spelling, puts its files
/// there, where DuckDB and this crate read them (#29 review: keyed by the name, they landed
/// under `main/Orders/` and `main/b/`, which no reader looks in).
#[tokio::test(flavor = "multi_thread")]
async fn a_write_into_a_renamed_table_lands_under_its_path() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data)
        .execute_batch(
            r#"CREATE TABLE lake.main.orders (id INT, name VARCHAR);
               INSERT INTO lake.main.orders VALUES (1, 'one');
               ALTER TABLE lake.main.orders RENAME TO "Orders";
               CREATE TABLE lake.main.a (id INT, name VARCHAR);
               INSERT INTO lake.main.a VALUES (10, 'ten');
               ALTER TABLE lake.main.a RENAME TO b;"#,
        )
        .unwrap();
    let writer = table_writer(&catalog).await;
    writer
        .append_table("main", "orders", &[batch(&[2])])
        .await
        .unwrap();
    writer
        .append_table("main", "ORDERS", &[batch(&[3])])
        .await
        .unwrap();
    writer
        .append_table("main", "B", &[batch(&[11])])
        .await
        .unwrap();
    assert_eq!(parquet_files(&data), ["main/a", "main/orders"]);

    let lake = duckdb(&catalog, &data);
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.main.orders ORDER BY id"),
        ["1", "2", "3"]
    );
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.main.b ORDER BY id"),
        ["10", "11"]
    );
    let (ctx, _) = crate_context(&catalog).await;
    assert_eq!(
        crate_answer(&ctx, "SELECT count(*) FROM lake.main.orders").await,
        Ok(vec!["3".to_string()])
    );
    assert_eq!(
        crate_answer(&ctx, "SELECT count(*) FROM lake.main.b").await,
        Ok(vec!["2".to_string()])
    );
}

/// A replace abandoned before its commit leaves its pending table row, spelled as it was
/// spelled. That row is no table's: a later write into the live table keys its files by the
/// live table's `path`, and a later write that takes the pending row up writes under that
/// row's `path` (#29 reviews 1 and 2).
#[tokio::test(flavor = "multi_thread")]
async fn a_pending_row_an_abandoned_write_left_steers_no_later_write() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data)
        .execute_batch(
            "CREATE TABLE lake.main.orders (id INT, name VARCHAR);
             INSERT INTO lake.main.orders VALUES (1, 'one');",
        )
        .unwrap();
    let writer = table_writer(&catalog).await;
    let pending_rows = |catalog: &Path| {
        rows(
            &duckdb(catalog, &data),
            "SELECT table_name || '|' || path FROM __ducklake_metadata_lake.ducklake_table
              WHERE begin_snapshot = 9223372036854775807 ORDER BY table_id",
        )
    };

    // A replace with a new spelling, abandoned; and a new table's create, abandoned.
    for (table, mode) in [("ORDERS", WriteMode::CreateOrReplace), ("NewT", WriteMode::Create)] {
        let mut session = writer
            .begin_write("main", table, &batch(&[]).schema(), mode)
            .unwrap();
        session.write_batch(&batch(&[7])).unwrap();
        // Abandoned as a process that dies abandons it: nothing of it runs again, so its
        // rows stay (a session dropped in-process releases them, #15).
        std::mem::forget(session);
    }
    assert_eq!(pending_rows(&catalog), ["ORDERS|ORDERS/", "NewT|NewT/"]);

    // The live table is written where it lives, not where the pending row would put it.
    writer
        .append_table("main", "Orders", &[batch(&[2])])
        .await
        .unwrap();
    assert_eq!(
        rows(
            &duckdb(&catalog, &data),
            "SELECT id FROM lake.main.orders ORDER BY id"
        ),
        ["1", "2"]
    );
    // A write that takes a pending row up makes it its own: spelling and path.
    writer
        .append_table("main", "newt", &[batch(&[20])])
        .await
        .unwrap();
    writer
        .create_or_replace_table("main", "OrDeRs", &[batch(&[5, 6])])
        .await
        .unwrap();

    let lake = duckdb(&catalog, &data);
    // A pending row is taken up as it is, spelled and placed as the write that made it left
    // it: nothing tells an abandoned row from one a write still in flight owns, and renaming
    // that one would move its files out from under it. DuckDB, which has no pending rows, would
    // store `OrDeRs` and `newt`; the spelling differs, every row is readable.
    assert_eq!(
        live_names(&lake),
        ["schema main", "table NewT", "table ORDERS"],
        "a pending row keeps its spelling"
    );
    assert_eq!(rows(&lake, "SELECT id FROM lake.main.newt"), ["20"]);
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.main.orders ORDER BY id"),
        ["5", "6"]
    );
    assert_eq!(
        as_this_filesystem_names(&data, parquet_files(&data)),
        as_this_filesystem_names(&data, ["main/NewT", "main/ORDERS", "main/orders"]),
        "nothing written under the abandoned spellings that a reader would not look in"
    );
}

/// Two writes in one transaction name one new schema in two spellings: the second finds the
/// first's pending row (in any ASCII case) and writes under its `path`, and nothing renames
/// it (#29 review 2: the second renamed the row to `staging/`, and the first's files, under
/// `Staging/a/`, were unreadable).
#[tokio::test(flavor = "multi_thread")]
async fn one_transaction_naming_a_new_schema_two_ways_writes_under_one_path() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data);
    let writer = table_writer(&catalog).await;
    let mut transaction = writer.transaction();
    let schema = batch(&[]).schema();
    transaction
        .stage_write("Staging", "a", &schema, WriteMode::Create, &[batch(&[1])])
        .await
        .unwrap();
    transaction
        .stage_write("staging", "b", &schema, WriteMode::Create, &[batch(&[2])])
        .await
        .unwrap();
    transaction.commit().await.unwrap();

    let lake = duckdb(&catalog, &data);
    assert_eq!(
        live_names(&lake),
        ["schema Staging", "schema main", "table a", "table b"]
    );
    assert_eq!(rows(&lake, "SELECT id FROM lake.staging.a"), ["1"]);
    assert_eq!(rows(&lake, "SELECT id FROM lake.STAGING.b"), ["2"]);
    assert_eq!(parquet_files(&data), ["Staging/a", "Staging/b"]);
}

/// Two writes in flight make one new table in two spellings: the second takes the first's
/// pending row up, and both write under its `path`. Whichever commits, its rows are readable
/// (#29 review 2: the second renamed the row to `newt/`, and the first's committed rows, under
/// `main/NewT/`, were not).
#[tokio::test(flavor = "multi_thread")]
async fn two_writes_in_flight_on_one_new_table_in_two_spellings_stay_readable() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data);
    let writer = table_writer(&catalog).await;
    let schema = batch(&[]).schema();
    let mut first = writer
        .begin_write("main", "NewT", &schema, WriteMode::Append)
        .unwrap();
    first.write_batch(&batch(&[1])).unwrap();
    let mut second = writer
        .begin_write("main", "newt", &schema, WriteMode::Append)
        .unwrap();
    second.write_batch(&batch(&[2])).unwrap();
    first.finish().await.unwrap();
    assert_eq!(
        rows(&duckdb(&catalog, &data), "SELECT id FROM lake.main.newt"),
        ["1"],
        "the first write's rows are readable"
    );
    // The second commits into the table the first published, or is refused; either way the
    // lake stays readable.
    let second = second.finish().await;
    let lake = duckdb(&catalog, &data);
    let expected: &[&str] = if second.is_ok() {
        &["1", "2"]
    } else {
        &["1"]
    };
    assert_eq!(
        rows(&lake, "SELECT id FROM lake.main.NEWT ORDER BY id"),
        expected
    );
    assert_eq!(live_names(&lake), ["schema main", "table NewT"]);
}

/// A write in flight into a new schema, and a second write that names it otherwise, begins
/// and is dropped: the first commits, and its rows are readable (#29 review 2: the dropped
/// write had renamed the pending schema to `staging/`).
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_write_naming_a_pending_schema_otherwise_moves_nothing() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data);
    let writer = table_writer(&catalog).await;
    let schema = batch(&[]).schema();
    let mut first = writer
        .begin_write("Staging", "x", &schema, WriteMode::Create)
        .unwrap();
    first.write_batch(&batch(&[1])).unwrap();
    let mut dropped = writer
        .begin_write("staging", "y", &schema, WriteMode::Create)
        .unwrap();
    dropped.write_batch(&batch(&[2])).unwrap();
    drop(dropped);
    first.finish().await.unwrap();

    let lake = duckdb(&catalog, &data);
    assert_eq!(rows(&lake, "SELECT id FROM lake.staging.x"), ["1"]);
    assert_eq!(rows(&lake, "SELECT id FROM lake.Staging.X"), ["1"]);
    assert_eq!(parquet_files(&data), ["Staging/x"]);
}

/// A `CREATE OR REPLACE` refuses, as a conflict, a table of its name another writer created
/// and dropped since it began, whatever spelling either used: DuckDB records the creation as
/// `created_table:"Staging"."T"`, which names what the replace spelled `"STAGING"."t"` (#29
/// review 2: compared exactly, the replace went ahead).
#[tokio::test(flavor = "multi_thread")]
async fn a_replace_conflicts_with_a_create_since_it_began_in_any_spelling() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = new_lake(&temp);
    duckdb(&catalog, &data)
        .execute_batch(r#"CREATE SCHEMA lake."Staging";"#)
        .unwrap();
    let writer = table_writer(&catalog).await;
    let mut replace = writer
        .begin_write(
            "STAGING",
            "t",
            &batch(&[]).schema(),
            WriteMode::CreateOrReplace,
        )
        .unwrap();
    replace.write_batch(&batch(&[1])).unwrap();
    duckdb(&catalog, &data)
        .execute_batch(
            r#"CREATE TABLE lake."Staging"."T" (id INT, name VARCHAR);
               DROP TABLE lake.staging.t;"#,
        )
        .unwrap();
    match replace.finish().await {
        Err(DuckLakeError::Conflict(message)) => {
            assert!(
                message.contains("has created a table with this name"),
                "{message}"
            )
        },
        other => panic!("a replace committed past a create since it began: {other:?}"),
    }
}
