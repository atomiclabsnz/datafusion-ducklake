//! `WriteMode::CreateOrReplace`, DuckDB's `CREATE OR REPLACE TABLE`, on the SQLite catalog and
//! on a host's SQLite, against official DuckLake (ferrion#611).
//!
//! In one snapshot the live table of the name is ended as `DROP TABLE` ends it and a new table,
//! with a new id and the new columns, is created with the rows. So a replace may change a
//! column's type and nullability, where `WriteMode::Replace` (`INSERT OVERWRITE`) keeps them.
//!
//! One lake is built by DuckDB. A copy of its catalog takes the statement from DuckDB 1.5.6 and
//! another from this crate, and the two catalogs must then hold the same rows, every table.
//! DuckDB then reads the crate's lake back, and goes on writing it.
//!
//! What is normalised, and why. Each side's own draws: a snapshot's time, the UUIDs, a Parquet
//! file's name and byte sizes (two Parquet writers). And what this crate's table creation writes
//! differently from DuckDB's in every mode — `WriteMode::Create` writes the same rows
//! (`no_prior_table` asserts it), so none of it is this mode's:
//! - its own `next_column_id` / `next_file_id` counters in `ducklake_metadata`;
//! - a table's column ids and orders, counted here from its first (DuckDB numbers each table's
//!   columns from 1; this crate draws ids from one counter across tables, and orders by id);
//! - a column's `default_value_type`/`default_value_dialect` (DuckDB writes `literal`/`duckdb`,
//!   this crate NULL), a decimal's spelling (`decimal(2, 1)` beside `decimal(2,1)`), a table's
//!   `table_uuid` (NULL here) and a data file's `file_format` (NULL here);
//! - the head snapshot's `next_catalog_id` / `next_file_id` watermarks: a write's start raises
//!   the head's past the pending table row it makes, and an inlined insert does not take a file
//!   id here as DuckDB's does. DuckDB going on writing the lake checks they stay past every id.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Decimal128Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeError, DuckLakeTableWriter, MetadataWriter, SqliteMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

fn duckdb(catalog: &Path, data: &Path, inlining: usize) -> duckdb::Connection {
    crate::common::ensure_ducklake_installed();
    crate::common::ensure_extension_installed("sqlite");
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite;
             ATTACH 'ducklake:sqlite:{}' AS lake (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT {inlining});",
            catalog.display(),
            data.display()
        ))
        .unwrap();
    connection
}

/// The metadata writer the crate writes through: the native SQLite one, or the same writer
/// over a host's SQLite (`SqlxHost`, the native connection behind the host interface).
async fn metadata_writer(catalog: &Path, inlining: usize, host: bool) -> Arc<dyn MetadataWriter> {
    let url = format!("sqlite:{}?mode=rw", catalog.display());
    if host {
        #[cfg(feature = "host-sqlite")]
        {
            use datafusion_ducklake::host_sqlite::{HostSqliteMetadataWriter, SqlxHost};
            let writer = HostSqliteMetadataWriter::from_host(SqlxHost::open(&url).unwrap())
                .unwrap()
                .with_data_inlining_row_limit(inlining);
            writer.initialize_schema().unwrap();
            return Arc::new(writer);
        }
        #[cfg(not(feature = "host-sqlite"))]
        unreachable!("host tests run with the host-sqlite feature");
    }
    let writer = SqliteMetadataWriter::new(&url)
        .await
        .unwrap()
        .with_data_inlining_row_limit(inlining);
    writer.initialize_schema().unwrap();
    Arc::new(writer)
}

fn table_writer(writer: Arc<dyn MetadataWriter>) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
}

/// Every row of every `ducklake_*` table, normalised as the module doc says.
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
    // A table's column ids counted from 1, as DuckDB numbers them.
    let first = "(SELECT min(f.column_id) FROM c.ducklake_column f WHERE f.table_id = x.table_id)";
    let column_id = format!("column_id - {first} + 1 AS column_id");
    let mut rows = Vec::new();
    for table in tables {
        let sql = match table.as_str() {
            "ducklake_snapshot" => {
                "SELECT * EXCLUDE (snapshot_time, next_catalog_id, next_file_id) FROM c.ducklake_snapshot x"
                    .to_string()
            },
            "ducklake_metadata" => {
                "SELECT * FROM c.ducklake_metadata x
                 WHERE key NOT IN ('next_column_id', 'next_file_id')"
                    .to_string()
            },
            "ducklake_data_file" => {
                "SELECT * EXCLUDE (path, file_size_bytes, footer_size, file_format)
                 FROM c.ducklake_data_file x"
                    .to_string()
            },
            "ducklake_table_stats" => {
                "SELECT * EXCLUDE (file_size_bytes) FROM c.ducklake_table_stats x".to_string()
            },
            "ducklake_table" => "SELECT * EXCLUDE (table_uuid) FROM c.ducklake_table x".to_string(),
            "ducklake_column" => format!(
                "SELECT * EXCLUDE (default_value_type, default_value_dialect)
                          REPLACE ({column_id},
                                   column_order - (SELECT min(f.column_order)
                                                   FROM c.ducklake_column f
                                                   WHERE f.table_id = x.table_id) + 1
                                       AS column_order,
                                   parent_column - {first} + 1 AS parent_column,
                                   replace(column_type, ', ', ',') AS column_type)
                 FROM c.ducklake_column x"
            ),
            "ducklake_file_column_stats" => format!(
                "SELECT * EXCLUDE (column_size_bytes) REPLACE ({column_id})
                 FROM c.ducklake_file_column_stats x"
            ),
            "ducklake_table_column_stats" | "ducklake_column_tag" => {
                format!("SELECT * REPLACE ({column_id}) FROM c.\"{table}\" x")
            },
            _ => format!("SELECT * FROM c.\"{table}\" x"),
        };
        let mut statement = connection
            .prepare(&format!("SELECT * FROM ({sql}) ORDER BY ALL"))
            .unwrap();
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

fn assert_same_catalog(official: &Path, ours: &Path) {
    let (official_rows, our_rows) = (dump(official), dump(ours));
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
}

/// `sql`'s rows as text, read by DuckDB.
fn rows(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = connection.prepare(sql).unwrap();
    let mut result = statement.query([]).unwrap();
    let mut rows = Vec::new();
    while let Some(row) = result.next().unwrap() {
        let columns = row.as_ref().column_count();
        rows.push(
            (0..columns)
                .map(|i| match row.get_ref(i).unwrap() {
                    duckdb::types::ValueRef::Text(text) => {
                        String::from_utf8_lossy(text).into_owned()
                    },
                    value => format!("{value:?}"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    rows
}

/// One `CREATE OR REPLACE TABLE`: the lake both sides start from, the statement DuckDB runs,
/// and the rows the crate writes for it.
struct Case {
    setup: &'static str,
    statement: &'static str,
    table: &'static str,
    batch: RecordBatch,
    inlining: usize,
}

/// Build the base lake, run the case on both sides, and compare the catalogs and what DuckDB
/// reads from each. Returns the crate's catalog.
async fn run(case: Case, host: bool) -> (TempDir, std::path::PathBuf) {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let base = temp.path().join("base.sqlite");
    duckdb(&base, &data, case.inlining)
        .execute_batch(case.setup)
        .unwrap();
    // The crate's counters seeded on the base, so both copies hold them.
    drop(metadata_writer(&base, case.inlining, false).await);

    let official = temp.path().join("official.sqlite");
    let ours = temp.path().join("ours.sqlite");
    std::fs::copy(&base, &official).unwrap();
    std::fs::copy(&base, &ours).unwrap();

    duckdb(&official, &data, case.inlining)
        .execute_batch(case.statement)
        .unwrap();
    let writer = table_writer(metadata_writer(&ours, case.inlining, host).await);
    writer
        .create_or_replace_table("main", case.table, &[case.batch])
        .await
        .unwrap();

    assert_same_catalog(&official, &ours);
    let read = format!("SELECT * FROM lake.main.{} ORDER BY ALL", case.table);
    let official_rows = rows(&duckdb(&official, &data, case.inlining), &read);
    let our_rows = rows(&duckdb(&ours, &data, case.inlining), &read);
    assert_eq!(
        our_rows, official_rows,
        "DuckDB reads the crate's table as its own"
    );

    // DuckDB goes on writing a copy of the crate's lake: the ids it allocates from the head's
    // watermarks collide with none the crate used.
    let after = temp.path().join("after.sqlite");
    std::fs::copy(&ours, &after).unwrap();
    let connection = duckdb(&after, &data, case.inlining);
    connection
        .execute_batch(&format!(
            "CREATE TABLE lake.main.later AS SELECT 1 AS x;
             INSERT INTO lake.main.later VALUES (2);
             CREATE OR REPLACE TABLE lake.main.{0} AS SELECT * FROM lake.main.{0};",
            case.table
        ))
        .unwrap();
    assert_eq!(
        rows(&connection, "SELECT x FROM lake.main.later ORDER BY x").len(),
        2
    );
    assert_eq!(
        rows(&connection, &read),
        official_rows,
        "and replaces it again"
    );
    (temp, ours)
}

/// `id` NULL as INTEGER and `d` 2.5 as DECIMAL(2,1): `CAST(NULL AS INTEGER) AS id, 2.5 AS d`.
fn null_id_and_decimal() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("d", DataType::Decimal128(2, 1), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![None::<i32>])) as ArrayRef,
            Arc::new(
                Decimal128Array::from(vec![Some(25_i128)])
                    .with_precision_and_scale(2, 1)
                    .unwrap(),
            ),
        ],
    )
    .unwrap()
}

fn bigint_and_varchar(id: i64, s: &str) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![id])) as ArrayRef,
            Arc::new(StringArray::from(vec![s])),
        ],
    )
    .unwrap()
}

/// A NOT NULL BIGINT table, with a row in Parquet and a comment, replaced by a nullable
/// INTEGER and a DECIMAL: the first of ferrion#611's failures (a NULL into the old NOT NULL
/// column was refused).
const NOT_NULL_BIGINT: &str = "
    CREATE TABLE lake.t (id BIGINT NOT NULL, s VARCHAR);
    INSERT INTO lake.t VALUES (1, 'a');
    COMMENT ON TABLE lake.t IS 'old';
";

fn changed_types_case(inlining: usize) -> Case {
    Case {
        setup: NOT_NULL_BIGINT,
        statement: "CREATE OR REPLACE TABLE lake.t AS SELECT CAST(NULL AS INTEGER) AS id, 2.5 AS d",
        table: "t",
        batch: null_id_and_decimal(),
        inlining,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn changed_types_and_nullability_in_parquet() {
    let (_temp, ours) = run(changed_types_case(0), false).await;
    let changes = rows(
        &duckdb::Connection::open_in_memory().unwrap(),
        &format!(
            "SELECT changes_made FROM sqlite_scan('{}', 'ducklake_snapshot_changes')
             ORDER BY snapshot_id DESC LIMIT 1",
            ours.display()
        ),
    );
    assert_eq!(
        changes,
        vec![r#"dropped_table:1,created_table:"main"."t",inserted_into_table:2"#]
    );
    // Before the replace, DuckDB still reads the old table.
    let data = ours.parent().unwrap().join("data");
    let connection = duckdb(&ours, &data, 0);
    assert_eq!(
        rows(&connection, "SELECT * FROM lake.main.t AT (VERSION => 3)"),
        vec!["BigInt(1)|a"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn changed_types_and_nullability_inlined() {
    run(changed_types_case(10), false).await;
}

/// ferrion#611's second failure: `CREATE OR REPLACE TABLE t (id BIGINT)` over an INTEGER `id`
/// was refused as a type change. No rows: the new table is created empty, with no data file
/// and no insert in the snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn a_new_type_with_no_rows() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
    run(
        Case {
            setup: "CREATE TABLE lake.t (id INTEGER); INSERT INTO lake.t VALUES (1), (2);",
            statement: "CREATE OR REPLACE TABLE lake.t (id BIGINT)",
            table: "t",
            batch: RecordBatch::new_empty(schema),
            inlining: 0,
        },
        false,
    )
    .await;
}

/// The same columns still make a new table, with a new id, as every DuckDB replace does.
#[tokio::test(flavor = "multi_thread")]
async fn an_identical_schema_is_a_new_table() {
    for inlining in [0, 10] {
        run(
            Case {
                setup: "CREATE TABLE lake.t (id BIGINT, s VARCHAR); INSERT INTO lake.t VALUES (1, 'a');",
                statement: "CREATE OR REPLACE TABLE lake.t AS SELECT 2::BIGINT AS id, 'b' AS s",
                table: "t",
                batch: bigint_and_varchar(2, "b"),
                inlining,
            },
            false,
        )
        .await;
    }
}

/// With no table of the name it is a create.
#[tokio::test(flavor = "multi_thread")]
async fn no_prior_table() {
    for inlining in [0, 10] {
        let (temp, ours) = run(
            Case {
                setup: "CREATE TABLE lake.other (i INTEGER); INSERT INTO lake.other VALUES (1);",
                statement: "CREATE OR REPLACE TABLE lake.t AS SELECT 2::BIGINT AS id, 'b' AS s",
                table: "t",
                batch: bigint_and_varchar(2, "b"),
                inlining,
            },
            false,
        )
        .await;
        // `WriteMode::Create` writes the very same catalog: what the dump normalises of this
        // crate's table creation is the create path's, not this mode's.
        let created = temp.path().join("created.sqlite");
        std::fs::copy(temp.path().join("base.sqlite"), &created).unwrap();
        table_writer(metadata_writer(&created, inlining, false).await)
            .create_table("main", "t", &[bigint_and_varchar(2, "b")])
            .await
            .unwrap();
        let raw = |catalog: &Path| {
            rows(
                &duckdb::Connection::open_in_memory().unwrap(),
                &format!(
                    "SELECT * FROM sqlite_scan('{0}', 'ducklake_column')
                     UNION ALL BY NAME
                     SELECT * EXCLUDE (table_uuid) FROM sqlite_scan('{0}', 'ducklake_table')
                     UNION ALL BY NAME
                     SELECT * EXCLUDE (snapshot_time) FROM sqlite_scan('{0}', 'ducklake_snapshot')
                     UNION ALL BY NAME
                     SELECT * FROM sqlite_scan('{0}', 'ducklake_snapshot_changes')
                     ORDER BY ALL",
                    catalog.display()
                ),
            )
        };
        assert_eq!(
            raw(&created),
            raw(&ours),
            "Create writes what CreateOrReplace does"
        );
    }
}

/// The old table's own `data_inlining_row_limit` stays with it: the new table, under the
/// writer's limit, inlines its row, as DuckDB's does.
#[tokio::test(flavor = "multi_thread")]
async fn the_old_tables_settings_stay_with_it() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let base = temp.path().join("base.sqlite");
    duckdb(&base, &data, 10)
        .execute_batch(NOT_NULL_BIGINT)
        .unwrap();
    // Set through the crate: the bundled engine's `set_option` bind crashes (see
    // standalone_ddl_tests), and the row it stores is the same.
    metadata_writer(&base, 10, false)
        .await
        .set_table_setting(1, "data_inlining_row_limit", "0")
        .unwrap();
    let official = temp.path().join("official.sqlite");
    let ours = temp.path().join("ours.sqlite");
    std::fs::copy(&base, &official).unwrap();
    std::fs::copy(&base, &ours).unwrap();
    duckdb(&official, &data, 10)
        .execute_batch(
            "CREATE OR REPLACE TABLE lake.t AS SELECT CAST(NULL AS INTEGER) AS id, 2.5 AS d",
        )
        .unwrap();
    table_writer(metadata_writer(&ours, 10, false).await)
        .create_or_replace_table("main", "t", &[null_id_and_decimal()])
        .await
        .unwrap();
    assert_same_catalog(&official, &ours);
}

/// A view of the name is refused, as DuckDB refuses it ("Existing object v is of type View"),
/// and the catalog is as it was: no snapshot, no table row, pending or not.
#[tokio::test(flavor = "multi_thread")]
async fn a_view_of_the_name_is_refused_and_nothing_is_written() {
    for inlining in [0, 10] {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let catalog = temp.path().join("c.sqlite");
        let connection = duckdb(&catalog, &data, inlining);
        connection
            .execute_batch("CREATE VIEW lake.v AS SELECT 1 AS one;")
            .unwrap();
        assert!(
            connection
                .execute_batch("CREATE OR REPLACE TABLE lake.v AS SELECT 1 AS id")
                .is_err(),
            "DuckDB refuses it"
        );
        drop(connection);
        let writer = table_writer(metadata_writer(&catalog, inlining, false).await);
        let before = std::fs::read(&catalog).unwrap();
        let files = walk(&data);
        let refused = writer
            .create_or_replace_table("main", "v", &[bigint_and_varchar(1, "a")])
            .await
            .expect_err("a view has the name");
        assert!(
            matches!(refused, DuckLakeError::ViewExists(_)),
            "{refused:?}"
        );
        assert!(
            std::fs::read(&catalog).unwrap() == before,
            "the catalog file is byte for byte as it was"
        );
        assert_eq!(walk(&data), files, "no file was written");
    }
}

/// Every file under `dir`.
fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// A table dropped, or another made under the name, by a commit after the replace began
/// refuses the replace, as DuckDB's conflict check does; the rows it would have written go.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_drop_or_create_of_the_name_is_a_conflict() {
    for concurrent in ["DROP TABLE lake.t", "CREATE TABLE lake.n (i INTEGER)"] {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let catalog = temp.path().join("c.sqlite");
        duckdb(&catalog, &data, 0)
            .execute_batch(NOT_NULL_BIGINT)
            .unwrap();
        let writer = table_writer(metadata_writer(&catalog, 0, false).await);
        let name = if concurrent.contains("lake.n") {
            "n"
        } else {
            "t"
        };
        let batch = null_id_and_decimal();
        let mut session = writer
            .begin_write(
                "main",
                name,
                batch.schema().as_ref(),
                WriteMode::CreateOrReplace,
            )
            .unwrap();
        session.write_batch(&batch).unwrap();
        duckdb(&catalog, &data, 0)
            .execute_batch(concurrent)
            .unwrap();
        let refused = session.finish().await.expect_err("a conflict");
        assert!(
            matches!(refused, DuckLakeError::Conflict(ref m) if m.contains("Transaction conflict")),
            "{concurrent}: {refused:?}"
        );
    }
}

/// `WriteMode::Replace` is unchanged: `INSERT OVERWRITE` keeps the table and its columns, so a
/// NULL into its NOT NULL column is still refused.
#[tokio::test(flavor = "multi_thread")]
async fn replace_keeps_the_table() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("c.sqlite");
    duckdb(&catalog, &data, 0)
        .execute_batch(NOT_NULL_BIGINT)
        .unwrap();
    let writer = table_writer(metadata_writer(&catalog, 0, false).await);
    assert!(
        writer
            .write_table("main", "t", &[null_id_and_decimal()])
            .await
            .is_err(),
        "the old columns still hold"
    );
    let written = writer
        .write_table("main", "t", &[bigint_and_varchar(2, "b")])
        .await
        .unwrap();
    assert_eq!(written.table_id, 1, "the same table");
}

/// Through a host's SQLite — the browser tab's catalog — the same catalog as official's.
#[cfg(feature = "host-sqlite")]
#[tokio::test(flavor = "multi_thread")]
async fn through_a_host_sqlite() {
    run(changed_types_case(0), true).await;
    run(changed_types_case(10), true).await;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
    run(
        Case {
            setup: "CREATE TABLE lake.t (id INTEGER); INSERT INTO lake.t VALUES (1), (2);",
            statement: "CREATE OR REPLACE TABLE lake.t (id BIGINT)",
            table: "t",
            batch: RecordBatch::new_empty(schema),
            inlining: 0,
        },
        true,
    )
    .await;
}

/// The backends that do not implement it refuse it by name before anything is written.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_catalog_refuses_it() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("c.duckdb");
    let writer = datafusion_ducklake::DuckdbMetadataWriter::new(catalog.to_str().unwrap()).unwrap();
    writer.initialize_schema().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let refused = table_writer(Arc::new(writer))
        .create_or_replace_table("main", "t", &[bigint_and_varchar(1, "a")])
        .await
        .expect_err("not implemented on DuckDB");
    assert!(
        matches!(refused, DuckLakeError::Unsupported(ref m) if m.contains("CREATE OR REPLACE")),
        "{refused:?}"
    );
}
