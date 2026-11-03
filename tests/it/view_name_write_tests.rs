//! A write onto a name a view holds, against official DuckLake (atomiclabsnz/datafusion-ducklake#17).
//!
//! DuckDB builds the lake and plays the other writer. A write that would make a table where a
//! view has the name is refused `ViewExists` at its start, before a file is written or a
//! catalog row changes; and a write whose table DuckDB drops and replaces with a view while it
//! runs is refused at its commit as a conflict, typed, not a missing catalog row. Either way
//! DuckDB then reads the catalog as it left it.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeError, DuckLakeTableWriter, MetadataWriter, SqliteMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use crate::standalone_ddl_tests::dump;

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

/// What DuckDB answers for `sql`, one line per row.
fn duckdb_rows(catalog: &Path, data: &Path, sql: &str) -> Vec<String> {
    let connection = duckdb(catalog, data);
    let mut statement = connection.prepare(sql).unwrap();
    let mut result = statement.query([]).unwrap();
    let mut rows = Vec::new();
    while let Some(row) = result.next().unwrap() {
        let columns = row.as_ref().column_count();
        let cells: Vec<String> = (0..columns)
            .map(|i| format!("{:?}", row.get_ref(i).unwrap()))
            .collect();
        rows.push(cells.join("|"));
    }
    rows
}

/// A table writer over the catalog; `inlining` 0 puts every row in Parquet. Opening it
/// brings DuckDB's catalog up to what this crate writes, once.
async fn table_writer(catalog: &Path, inlining: usize) -> DuckLakeTableWriter {
    let url = format!("sqlite:{}?mode=rw", catalog.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer.with_data_inlining_row_limit(inlining));
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
}

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ])
}

/// `rows` rows: more than any inlining limit here, so a write that went ahead would stage
/// Parquet.
fn batch(rows: i32) -> RecordBatch {
    let ids: Vec<i32> = (0..rows).collect();
    let names: Vec<String> = ids.iter().map(|id| format!("row {id}")).collect();
    RecordBatch::try_new(
        Arc::new(schema()),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

/// Every file under `dir`.
fn files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, found);
            } else {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, &mut found);
    found.sort();
    found
}

fn assert_view_exists(error: &DuckLakeError, name: &str) {
    match error {
        DuckLakeError::ViewExists(message) => assert!(
            message.starts_with(&format!("{name}:")),
            "the refusal names the view {name}: {message}"
        ),
        _ => panic!("a table made where a view has its name is ViewExists: {error:?}"),
    }
}

/// The lake DuckDB builds: a table `t` with rows in Parquet, and a view `v`.
fn lake(temp: &TempDir) -> (PathBuf, PathBuf) {
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("catalog.sqlite");
    duckdb(&catalog, &data)
        .execute_batch(
            "CREATE TABLE lake.main.t (id INT, name VARCHAR);
             INSERT INTO lake.main.t VALUES (1, 'one'), (2, 'two');
             CREATE VIEW lake.main.v AS SELECT 7 AS id, 'seven' AS name;",
        )
        .unwrap();
    (catalog, data)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_onto_a_views_name_is_refused_before_any_file_is_written() {
    let temp = TempDir::new().unwrap();
    let (catalog, data) = lake(&temp);
    table_writer(&catalog, 0).await;
    let catalog_before = dump(&catalog);
    let files_before = files(&data);
    let view_before = duckdb_rows(&catalog, &data, "SELECT * FROM lake.main.v");
    assert_eq!(view_before.len(), 1);

    for inlining in [0, 10] {
        let writer = table_writer(&catalog, inlining).await;
        // Every mode would make the table here: there is none of the name.
        for mode in [WriteMode::Create, WriteMode::Replace, WriteMode::Append] {
            let error = writer
                .begin_write("main", "v", &schema(), mode)
                .err()
                .unwrap_or_else(|| panic!("{mode:?} inlining {inlining}: begun over a view"));
            assert_view_exists(&error, "main.v");
        }
        for (what, result) in [
            (
                "create_table",
                writer.create_table("main", "v", &[batch(100)]).await,
            ),
            (
                "write_table",
                writer.write_table("main", "v", &[batch(100)]).await,
            ),
            (
                "append_table",
                writer.append_table("main", "v", &[batch(100)]).await,
            ),
        ] {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{what} inlining {inlining}: wrote over a view"));
            assert_view_exists(&error, "main.v");
        }
        // Nothing was staged, uploaded or written to the catalog: not even the pending table
        // row a start makes for a new name.
        assert_eq!(files(&data), files_before, "inlining {inlining}");
        assert_eq!(dump(&catalog), catalog_before, "inlining {inlining}");
    }
    // DuckDB reads the lake as it was.
    assert_eq!(
        duckdb_rows(&catalog, &data, "SELECT * FROM lake.main.v"),
        view_before
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_dropped_and_replaced_by_a_view_during_a_write_is_a_typed_conflict() {
    for mode in [WriteMode::Append, WriteMode::Replace] {
        for inlining in [0, 10] {
            let temp = TempDir::new().unwrap();
            let (catalog, data) = lake(&temp);
            let writer = table_writer(&catalog, inlining).await;

            // A write begins on the live table, then DuckDB drops it and makes a view of its
            // name before the write commits.
            let mut late = writer.begin_write("main", "t", &schema(), mode).unwrap();
            late.write_batch(&batch(3)).unwrap();
            duckdb(&catalog, &data)
                .execute_batch(
                    "DROP TABLE lake.main.t;
                     CREATE VIEW lake.main.t AS SELECT 42 AS id, 'view' AS name;",
                )
                .unwrap();
            let catalog_before = dump(&catalog);
            let error = late.finish().await.unwrap_err();
            assert!(
                matches!(error, DuckLakeError::Conflict(_)),
                "{mode:?} inlining {inlining}: a table dropped under a write is a conflict: \
                 {error:?}"
            );
            assert_eq!(
                dump(&catalog),
                catalog_before,
                "{mode:?} inlining {inlining}: the refused commit changed the catalog"
            );
            // DuckDB reads the view it made.
            assert_eq!(
                duckdb_rows(&catalog, &data, "SELECT * FROM lake.main.t"),
                duckdb_rows(&catalog, &data, "SELECT 42 AS id, 'view' AS name"),
                "{mode:?} inlining {inlining}"
            );
            // A retry begins on the catalog as it now is: the view's name, refused at once.
            let error = writer
                .append_table("main", "t", &[batch(3)])
                .await
                .unwrap_err();
            assert_view_exists(&error, "main.t");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_dropped_during_a_write_is_a_typed_conflict() {
    // The same race without the view: still a conflict, not a missing catalog row.
    let temp = TempDir::new().unwrap();
    let (catalog, data) = lake(&temp);
    let writer = table_writer(&catalog, 0).await;
    let mut late = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    late.write_batch(&batch(3)).unwrap();
    duckdb(&catalog, &data)
        .execute_batch("DROP TABLE lake.main.t;")
        .unwrap();
    let error = late.finish().await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::Conflict(_)),
        "a table dropped under a write is a conflict: {error:?}"
    );
    assert_eq!(
        duckdb_rows(
            &catalog,
            &data,
            "SELECT COUNT(*) FROM duckdb_tables() WHERE database_name = 'lake'"
        ),
        ["BigInt(0)"]
    );
}
