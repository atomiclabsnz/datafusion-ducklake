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
