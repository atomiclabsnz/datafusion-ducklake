//! The catalog rows a table's creation writes, against DuckDB 1.5.6's (ducklake `ac7595b0`),
//! row by row, in every write mode, on the SQLite catalog and a host's SQLite
//! (atomiclabsnz/datafusion-ducklake#27).
//!
//! DuckDB builds one lake. A copy of its catalog takes a statement from DuckDB and another the
//! same change from this crate, and the two catalogs must then hold the same rows. Only each
//! side's own draws are normalised: a snapshot's time, a UUID's value (not whether there is
//! one), a Parquet file's name and byte sizes (two Parquet writers). Everything else is compared
//! as written — column ids and orders, types, defaults, `file_format`, `table_uuid`, the
//! snapshot changes — but for these residuals, each pinned where it is normalised:
//! - this crate's own allocator rows in `ducklake_metadata` (`next_column_id`, `next_file_id`,
//!   ...), which DuckDB does not read;
//! - the order of a snapshot's changes (`altered_table:2,inserted_into_table:2` beside DuckDB's
//!   `inserted_into_table:2,altered_table:2`), which no reader depends on;
//! - the text of a nested value in an inlined data table: DuckDB writes `[a, b]` where this
//!   crate writes `['a', 'b']`, and DuckDB reads both as the same list;
//! - the snapshots' `next_catalog_id` / `next_file_id`, compared apart ([`assert_watermarks`]):
//!   the head's `next_catalog_id` is DuckDB's; a write's start raises the snapshot it started
//!   at past the pending table row it makes, so no DuckDB writer meanwhile takes the row's id;
//!   and an inlined insert takes no file id here, where DuckDB's takes one per table.
//!
//! Then each side goes on writing the other's lake — DuckDB the crate's, the crate DuckDB's —
//! and no id is used twice: a catalog id names one object, a column id one column of its table,
//! a file id one file. Every file's Parquet field ids are its table's column ids, and DuckDB reads
//! every column by them after the schema changes.
//!
//! The DuckDB catalog is not compared here: this crate's DuckDB writer writes a catalog it made
//! (`DuckdbMetadataWriter::new_with_init`), which DuckDB does not write, and DuckDB's, which the
//! writer does not; `a_duckdb_catalog_numbers_and_types_as_duckdb_does` checks its rows.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite", feature = "metadata-duckdb"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Decimal128Array, Int32Array, Int64Array, ListArray, StringArray, StructArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    ColumnDef, DuckLakeCatalog, DuckLakeTableWriter, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

/// The writer the crate writes the SQLite catalog through.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Backend {
    Sqlite,
    /// The same writer over a host's SQLite (`SqlxHost`, the native connection behind the host
    /// interface).
    #[cfg(feature = "host-sqlite")]
    HostSqlite,
}

/// One DuckDB-made lake and the copies of its catalog the two sides write.
struct Lake {
    temp: TempDir,
    data: PathBuf,
    backend: Backend,
    inlining: usize,
}

impl Lake {
    /// DuckDB makes the lake and runs `setup` in it; the crate then opens it once, so its own
    /// allocator rows are in every copy.
    async fn new(backend: Backend, inlining: usize, setup: &str) -> Self {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let lake = Self {
            temp,
            data,
            backend,
            inlining,
        };
        let base = lake.file("base");
        lake.duckdb(&base).execute_batch(setup).unwrap();
        drop(lake.writer(&base).await);
        lake
    }

    fn file(&self, name: &str) -> PathBuf {
        self.temp.path().join(format!("{name}.sqlite"))
    }

    /// A copy of the base catalog named `name`.
    fn copy(&self, name: &str) -> PathBuf {
        let path = self.file(name);
        std::fs::copy(self.file("base"), &path).unwrap();
        path
    }

    /// DuckDB with `catalog` attached as `lake`.
    fn duckdb(&self, catalog: &Path) -> duckdb::Connection {
        crate::common::ensure_ducklake_installed();
        crate::common::ensure_extension_installed("sqlite");
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD ducklake; LOAD sqlite;
                 ATTACH 'ducklake:sqlite:{}' AS lake
                     (DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT {});",
                catalog.display(),
                self.data.display(),
                self.inlining
            ))
            .unwrap();
        connection
    }

    /// This crate's metadata writer over `catalog`.
    async fn writer(&self, catalog: &Path) -> Arc<dyn MetadataWriter> {
        let url = format!("sqlite:{}?mode=rw", catalog.display());
        match self.backend {
            Backend::Sqlite => {
                let writer = SqliteMetadataWriter::new(&url)
                    .await
                    .unwrap()
                    .with_data_inlining_row_limit(self.inlining);
                writer.initialize_schema().unwrap();
                Arc::new(writer)
            },
            #[cfg(feature = "host-sqlite")]
            Backend::HostSqlite => {
                use datafusion_ducklake::host_sqlite::{HostSqliteMetadataWriter, SqlxHost};
                let writer = HostSqliteMetadataWriter::from_host(SqlxHost::open(&url).unwrap())
                    .unwrap()
                    .with_data_inlining_row_limit(self.inlining);
                writer.initialize_schema().unwrap();
                Arc::new(writer)
            },
        }
    }

    fn table_writer(&self, writer: Arc<dyn MetadataWriter>) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
    }

    /// A DataFusion session with `catalog` registered, writable, as `ducklake`.
    async fn session(&self, catalog: &Path) -> SessionContext {
        let writer = self.writer(catalog).await;
        let provider =
            SqliteMetadataProvider::new(&format!("sqlite:{}?mode=rw", catalog.display()))
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "ducklake",
            Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), writer).unwrap()),
        );
        ctx
    }

    /// DuckDB with `catalog`'s SQLite database attached, read-only, as `c`.
    fn catalog_database(catalog: &Path) -> duckdb::Connection {
        crate::common::ensure_extension_installed("sqlite");
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD sqlite; ATTACH '{}' AS c (TYPE sqlite, READ_ONLY);",
                catalog.display()
            ))
            .unwrap();
        connection
    }

    /// Every row of every `ducklake_*` table of `catalog`, normalised as the module doc says.
    fn dump(&self, catalog: &Path) -> Vec<String> {
        let connection = Self::catalog_database(catalog);
        let tables = texts(
            &connection,
            "SELECT table_name FROM information_schema.tables
             WHERE table_catalog = 'c' AND table_name LIKE 'ducklake%' ORDER BY table_name",
        );
        let uuid =
            regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
                .unwrap();
        let mut dumped = Vec::new();
        for table in tables {
            let sql = match table.as_str() {
                // The watermarks are compared apart.
                "ducklake_snapshot" => "SELECT * EXCLUDE (snapshot_time, next_catalog_id,
                                                      next_file_id)
                                        FROM c.ducklake_snapshot"
                    .to_string(),
                "ducklake_metadata" => "SELECT * FROM c.ducklake_metadata
                     WHERE key NOT IN ('next_column_id', 'next_file_id', 'next_partition_id',
                                       'next_sort_id')"
                    .to_string(),
                "ducklake_snapshot_changes" => "SELECT * REPLACE (
                         array_to_string(list_sort(string_split(changes_made, ',')), ',')
                             AS changes_made)
                     FROM c.ducklake_snapshot_changes"
                    .to_string(),
                "ducklake_data_file" | "ducklake_delete_file" => {
                    format!("SELECT * EXCLUDE (path, file_size_bytes, footer_size) FROM c.{table}")
                },
                "ducklake_table_stats" => {
                    "SELECT * EXCLUDE (file_size_bytes) FROM c.ducklake_table_stats".to_string()
                },
                "ducklake_file_column_stats" => {
                    "SELECT * EXCLUDE (column_size_bytes) FROM c.ducklake_file_column_stats"
                        .to_string()
                },
                _ => format!("SELECT * FROM c.\"{table}\""),
            };
            for row in rows(&connection, &format!("SELECT * FROM ({sql}) ORDER BY ALL")) {
                let row = if table.starts_with("ducklake_inlined_data_") {
                    row.replace('\'', "")
                } else {
                    row
                };
                dumped.push(
                    uuid.replace_all(&format!("{table} {row}"), "<uuid>")
                        .into_owned(),
                );
            }
        }
        dumped
    }

    /// The rows only one of `official` and `ours` holds: `- ` official's, `+ ` ours.
    fn diff(&self, official: &Path, ours: &Path) -> Vec<String> {
        let (official, ours) = (self.dump(official), self.dump(ours));
        let mut diff: Vec<String> = official
            .iter()
            .filter(|row| !ours.contains(row))
            .map(|row| format!("- {row}"))
            .collect();
        diff.extend(
            ours.iter()
                .filter(|row| !official.contains(row))
                .map(|row| format!("+ {row}")),
        );
        diff
    }

    /// The watermarks residual (module doc), pinned: the same snapshots; the head's
    /// `next_catalog_id` DuckDB's; an earlier snapshot's no lower than DuckDB's; and the head's
    /// `next_file_id` DuckDB's, less one for each commit that inlined rows into a table.
    fn assert_watermarks(&self, official: &Path, ours: &Path) {
        let watermarks = |catalog: &Path| -> Vec<(i64, i64, i64)> {
            let connection = Self::catalog_database(catalog);
            let mut statement = connection
                .prepare(
                    "SELECT snapshot_id, next_catalog_id, next_file_id FROM c.ducklake_snapshot
                     ORDER BY snapshot_id",
                )
                .unwrap();
            statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        };
        let inlined_inserts = |catalog: &Path| -> i64 {
            let connection = Self::catalog_database(catalog);
            rows(
                &connection,
                "SELECT count(*) FROM c.ducklake_snapshot_changes,
                     unnest(string_split(changes_made, ',')) AS change(c)
                 WHERE c LIKE 'inlined_insert:%'",
            )[0]
            .parse()
            .unwrap()
        };
        let (theirs, mine) = (watermarks(official), watermarks(ours));
        let ids = |marks: &[(i64, i64, i64)]| marks.iter().map(|m| m.0).collect::<Vec<_>>();
        assert_eq!(ids(&mine), ids(&theirs), "the same snapshots");
        let (head_theirs, head_mine) = (theirs.last().unwrap(), mine.last().unwrap());
        assert_eq!(head_mine.1, head_theirs.1, "the head's next_catalog_id");
        for (t, m) in theirs.iter().zip(&mine) {
            assert!(
                m.1 >= t.1,
                "snapshot {}'s next_catalog_id {} < DuckDB's {}",
                t.0,
                m.1,
                t.1
            );
        }
        let base = self.file("base");
        let new_inlined = inlined_inserts(official) - inlined_inserts(&base);
        assert_eq!(
            head_mine.2,
            head_theirs.2 - new_inlined,
            "the head's next_file_id, one fewer per inlined insert"
        );
    }

    /// `catalog`'s ids each name one thing, after both sides wrote it: a catalog id one
    /// schema, table or view; a column id one column of its table; a file id one file.
    fn assert_ids_distinct(&self, catalog: &Path) {
        let connection = self.duckdb(catalog);
        let reused = rows(
            &connection,
            "WITH m AS (SELECT * FROM __ducklake_metadata_lake.ducklake_schema),
                  catalog_ids AS (
                      SELECT schema_id AS id, 'schema ' || schema_name AS name FROM m
                      UNION ALL
                      SELECT table_id, 'table ' || table_name
                      FROM __ducklake_metadata_lake.ducklake_table
                      UNION ALL
                      SELECT view_id, 'view ' || view_name
                      FROM __ducklake_metadata_lake.ducklake_view)
             SELECT 'catalog id ' || id FROM catalog_ids GROUP BY id
             HAVING count(DISTINCT name) > 1
             UNION ALL
             SELECT 'column id ' || table_id || '.' || column_id
             FROM __ducklake_metadata_lake.ducklake_column GROUP BY table_id, column_id
             HAVING count(DISTINCT column_name) > 1
             UNION ALL
             SELECT 'file id ' || data_file_id
             FROM __ducklake_metadata_lake.ducklake_data_file GROUP BY data_file_id
             HAVING count(*) > 1
             UNION ALL
             SELECT 'delete file id ' || delete_file_id
             FROM __ducklake_metadata_lake.ducklake_delete_file GROUP BY delete_file_id
             HAVING count(*) > 1",
        );
        assert_eq!(reused, Vec::<String>::new(), "an id names two things");
    }
}

/// The one text column of `sql`'s rows, as they are.
fn texts(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// DuckDB's answer to `sql`, one `|`-joined line per row, a text quoted.
fn rows(connection: &duckdb::Connection, sql: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut result = statement.query([]).unwrap();
    let mut lines = Vec::new();
    while let Some(row) = result.next().unwrap() {
        let columns = row.as_ref().column_count();
        let cells: Vec<String> = (0..columns)
            .map(|i| match row.get_ref(i).unwrap() {
                duckdb::types::ValueRef::Null => "NULL".to_string(),
                duckdb::types::ValueRef::Text(text) => {
                    format!("{:?}", String::from_utf8_lossy(text))
                },
                duckdb::types::ValueRef::Int(value) => value.to_string(),
                duckdb::types::ValueRef::BigInt(value) => value.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        lines.push(cells.join("|"));
    }
    lines
}

/// The lake every case starts from: one table DuckDB made, with a row in Parquet, so ids the
/// case draws are not the first of their kind.
const BASE: &str = "CREATE TABLE lake.other (i INTEGER, j VARCHAR);
                    INSERT INTO lake.other VALUES (1, 'a');";

/// `id BIGINT, d DECIMAL(2,1), s VARCHAR`, one row: `(1, 2.5, 'x')`.
const FLAT_CTAS: &str = "1::BIGINT AS id, 2.5::DECIMAL(2,1) AS d, 'x' AS s";

fn flat(id: i64, d: i128, s: &str) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("d", DataType::Decimal128(2, 1), true),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![id])) as ArrayRef,
            Arc::new(
                Decimal128Array::from(vec![d])
                    .with_precision_and_scale(2, 1)
                    .unwrap(),
            ),
            Arc::new(StringArray::from(vec![s])),
        ],
    )
    .unwrap()
}

/// `n STRUCT(x INTEGER, y VARCHAR[]), l INTEGER[]`, one row.
const NESTED_CTAS: &str =
    "{'x': 1, 'y': ['a', 'b']}::STRUCT(x INTEGER, y VARCHAR[]) AS n, [1, 2]::INTEGER[] AS l";

fn nested() -> RecordBatch {
    let element = Arc::new(Field::new("element", DataType::Utf8, true));
    let y = ListArray::new(
        Arc::clone(&element),
        OffsetBuffer::new(vec![0, 2].into()),
        Arc::new(StringArray::from(vec!["a", "b"])),
        None,
    );
    let struct_fields = Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::List(element), true),
    ]);
    let n = StructArray::new(
        struct_fields.clone(),
        vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef, Arc::new(y)],
        None,
    );
    let int_element = Arc::new(Field::new("element", DataType::Int32, true));
    let l = ListArray::new(
        Arc::clone(&int_element),
        OffsetBuffer::new(vec![0, 2].into()),
        Arc::new(Int32Array::from(vec![1, 2])),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Struct(struct_fields), true),
        Field::new("l", DataType::List(int_element), true),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(n) as ArrayRef, Arc::new(l)]).unwrap()
}

/// How the crate makes the table.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Create,
    CreateOrReplace,
    /// An append to a table that does not exist yet.
    AppendIntoNew,
}

impl Mode {
    async fn write(self, writer: &DuckLakeTableWriter, table: &str, batch: RecordBatch) {
        match self {
            Self::Create => writer.create_table("main", table, &[batch]).await,
            Self::CreateOrReplace => {
                writer
                    .create_or_replace_table("main", table, &[batch])
                    .await
            },
            Self::AppendIntoNew => writer.append_table("main", table, &[batch]).await,
        }
        .unwrap();
    }
}

fn backends() -> Vec<Backend> {
    vec![
        Backend::Sqlite,
        #[cfg(feature = "host-sqlite")]
        Backend::HostSqlite,
    ]
}

const MODES: [Mode; 3] = [Mode::Create, Mode::CreateOrReplace, Mode::AppendIntoNew];

/// The catalog diff, reported with what made it.
fn assert_no_diff(lake: &Lake, what: &str, official: &Path, ours: &Path) {
    let diff = lake.diff(official, ours);
    assert!(
        diff.is_empty(),
        "{what} ({:?}, inlining {}): the catalogs differ\n{}",
        lake.backend,
        lake.inlining,
        diff.join("\n")
    );
    lake.assert_watermarks(official, ours);
}

/// `flat` with a fourth column, `z INTEGER`.
fn flat_with_z(id: i64, d: i128, s: &str, z: i32) -> RecordBatch {
    let base = flat(id, d, s);
    let mut fields = base.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("z", DataType::Int32, true)));
    let mut columns = base.columns().to_vec();
    columns.push(Arc::new(Int32Array::from(vec![z])));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Every data file's Parquet field ids are its table's column ids: each `(name, field_id)` of
/// a file's schema is a column of the table with that id.
fn assert_field_ids_are_column_ids(lake: &Lake, catalog: &Path) {
    let connection = lake.duckdb(catalog);
    let tables = texts(
        &connection,
        "SELECT table_id || '|' || table_name FROM __ducklake_metadata_lake.ducklake_table
         WHERE end_snapshot IS NULL",
    );
    for table in tables {
        let (table_id, name) = table.split_once('|').unwrap();
        let files = texts(
            &connection,
            &format!("SELECT data_file FROM ducklake_list_files('lake', '{name}')"),
        );
        for file in files {
            let unknown = rows(
                &connection,
                &format!(
                    "SELECT name || ' ' || field_id FROM parquet_schema('{file}')
                     WHERE field_id IS NOT NULL
                     EXCEPT
                     SELECT column_name || ' ' || column_id
                     FROM __ducklake_metadata_lake.ducklake_column WHERE table_id = {table_id}"
                ),
            );
            assert_eq!(
                unknown,
                Vec::<String>::new(),
                "{file} of {name}: field ids that are no column id of the table"
            );
        }
    }
}

/// DuckDB's script after the case, on a copy of `catalog`: a new table, a column `z` added to
/// `t` and a row with it.
fn duckdb_goes_on(lake: &Lake, catalog: &Path, name: &str) -> PathBuf {
    let after = lake.file(name);
    std::fs::copy(catalog, &after).unwrap();
    lake.duckdb(&after)
        .execute_batch(
            "CREATE TABLE lake.later AS SELECT 1 AS x;
             INSERT INTO lake.later VALUES (2);
             ALTER TABLE lake.t ADD COLUMN z INTEGER;
             INSERT INTO lake.t VALUES (2, 3.5, 'y', 9);",
        )
        .unwrap();
    after
}

/// The crate's equivalent on a copy of `catalog`: a new table, and an append to `t` with a new
/// column `z`.
async fn crate_goes_on(lake: &Lake, catalog: &Path, name: &str) -> PathBuf {
    let after = lake.file(name);
    std::fs::copy(catalog, &after).unwrap();
    let writer = lake.table_writer(lake.writer(&after).await);
    let x = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
    )
    .unwrap();
    writer.create_table("main", "later", &[x]).await.unwrap();
    writer
        .append_table("main", "t", &[flat_with_z(2, 35, "y", 9)])
        .await
        .unwrap();
    after
}

/// Each side goes on writing both lakes (`t` must be the flat table): no id is used twice,
/// every file's field ids are its table's column ids, and DuckDB reads `t` the same from the
/// crate's lake as from its own.
async fn write_each_others(lake: &Lake, official: &Path, ours: &Path) {
    let read = |catalog: &Path| {
        rows(
            &lake.duckdb(catalog),
            "SELECT * FROM lake.t ORDER BY ALL NULLS FIRST",
        )
    };
    for (who, theirs, mine) in [
        (
            "DuckDB",
            duckdb_goes_on(lake, official, "duckdb_on_official"),
            duckdb_goes_on(lake, ours, "duckdb_on_ours"),
        ),
        (
            "the crate",
            crate_goes_on(lake, official, "crate_on_official").await,
            crate_goes_on(lake, ours, "crate_on_ours").await,
        ),
    ] {
        for catalog in [&theirs, &mine] {
            lake.assert_ids_distinct(catalog);
            assert_field_ids_are_column_ids(lake, catalog);
        }
        assert_eq!(
            read(&mine),
            read(&theirs),
            "{who} writing the crate's lake, then DuckDB reading it, as its own"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_with_rows_in_every_mode() {
    for backend in backends() {
        for inlining in [0, 10] {
            for mode in MODES {
                let lake = Lake::new(backend, inlining, BASE).await;
                let (official, ours) = (lake.copy("official"), lake.copy("ours"));
                lake.duckdb(&official)
                    .execute_batch(&format!("CREATE TABLE lake.t AS SELECT {FLAT_CTAS}"))
                    .unwrap();
                mode.write(
                    &lake.table_writer(lake.writer(&ours).await),
                    "t",
                    flat(1, 25, "x"),
                )
                .await;
                assert_no_diff(&lake, &format!("{mode:?}"), &official, &ours);
                write_each_others(&lake, &official, &ours).await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn nested_types() {
    for backend in backends() {
        for inlining in [0, 10] {
            for mode in MODES {
                let lake = Lake::new(backend, inlining, BASE).await;
                let (official, ours) = (lake.copy("official"), lake.copy("ours"));
                lake.duckdb(&official)
                    .execute_batch(&format!("CREATE TABLE lake.n AS SELECT {NESTED_CTAS}"))
                    .unwrap();
                mode.write(&lake.table_writer(lake.writer(&ours).await), "n", nested())
                    .await;
                assert_no_diff(&lake, &format!("nested {mode:?}"), &official, &ours);
                assert_eq!(
                    rows(&lake.duckdb(&ours), "SELECT * FROM lake.n"),
                    rows(&lake.duckdb(&official), "SELECT * FROM lake.n"),
                );
            }
        }
    }
}

/// Two tables, one after the other: each numbers its columns from 1, as DuckDB's do.
#[tokio::test(flavor = "multi_thread")]
async fn two_tables_in_one_lake() {
    for backend in backends() {
        for inlining in [0, 10] {
            let lake = Lake::new(backend, inlining, BASE).await;
            let (official, ours) = (lake.copy("official"), lake.copy("ours"));
            lake.duckdb(&official)
                .execute_batch(&format!(
                    "CREATE TABLE lake.t AS SELECT {FLAT_CTAS};
                     CREATE TABLE lake.n AS SELECT {NESTED_CTAS};"
                ))
                .unwrap();
            let writer = lake.table_writer(lake.writer(&ours).await);
            Mode::Create.write(&writer, "t", flat(1, 25, "x")).await;
            Mode::Create.write(&writer, "n", nested()).await;
            assert_no_diff(&lake, "two tables", &official, &ours);
            write_each_others(&lake, &official, &ours).await;
        }
    }
}

/// `CREATE TABLE t (columns)` and an empty `CREATE TABLE … AS` through DataFusion, which
/// reach `register_table`: no rows, so no insert in the snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn sql_create_table_with_no_rows() {
    for backend in backends() {
        for (duckdb_sql, datafusion_sql) in [
            (
                "CREATE TABLE lake.t (id BIGINT, d DECIMAL(2,1), s VARCHAR)",
                "CREATE TABLE ducklake.main.t (id BIGINT, d DECIMAL(2,1), s VARCHAR)",
            ),
            (
                "CREATE TABLE lake.t AS SELECT * FROM lake.other WHERE false",
                "CREATE TABLE ducklake.main.t AS SELECT * FROM ducklake.main.other WHERE false",
            ),
        ] {
            let lake = Lake::new(backend, 0, BASE).await;
            let (official, ours) = (lake.copy("official"), lake.copy("ours"));
            lake.duckdb(&official).execute_batch(duckdb_sql).unwrap();
            {
                let ctx = lake.session(&ours).await;
                ctx.sql(datafusion_sql)
                    .await
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
            }
            assert_no_diff(&lake, datafusion_sql, &official, &ours);
        }
    }
}

/// `get_or_create_table` then `set_columns`, with a column default.
#[tokio::test(flavor = "multi_thread")]
async fn set_columns_with_a_default() {
    for backend in backends() {
        let lake = Lake::new(backend, 0, BASE).await;
        let (official, ours) = (lake.copy("official"), lake.copy("ours"));
        lake.duckdb(&official)
            .execute_batch("CREATE TABLE lake.t (id BIGINT, d DECIMAL(2,1), k INTEGER DEFAULT 5)")
            .unwrap();
        {
            let writer = lake.writer(&ours).await;
            let snapshot = writer.create_snapshot().unwrap();
            let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
            let (table_id, _) = writer
                .get_or_create_table(schema_id, "t", None, snapshot)
                .unwrap();
            writer
                .set_columns(
                    table_id,
                    &[
                        ColumnDef::new("id", "int64", true).unwrap(),
                        ColumnDef::new("d", "decimal(2,1)", true).unwrap(),
                        ColumnDef::new("k", "int32", true)
                            .unwrap()
                            .with_default("5")
                            .unwrap(),
                    ],
                    snapshot,
                )
                .unwrap();
        }
        assert_no_diff(&lake, "set_columns", &official, &ours);
    }
}

/// A table both sides made the same, then a column added with a row, in one snapshot: the new
/// column takes the table's next id, as DuckDB's `ALTER TABLE … ADD COLUMN` gives it.
#[tokio::test(flavor = "multi_thread")]
async fn schema_evolution() {
    for backend in backends() {
        for inlining in [0, 10] {
            let lake = Lake::new(
                backend,
                inlining,
                // A wider table after `t`, so the catalog's highest column id is not `t`'s.
                &format!(
                    "{BASE} CREATE TABLE lake.t AS SELECT {FLAT_CTAS};
                     CREATE TABLE lake.wide (a INT, b INT, c INT, d INT, e INT, f INT);"
                ),
            )
            .await;
            let (official, ours) = (lake.copy("official"), lake.copy("ours"));
            lake.duckdb(&official)
                .execute_batch(
                    "BEGIN;
                     ALTER TABLE lake.t ADD COLUMN z INTEGER;
                     INSERT INTO lake.t VALUES (2, 3.5, 'y', 9);
                     COMMIT;",
                )
                .unwrap();
            lake.table_writer(lake.writer(&ours).await)
                .append_table("main", "t", &[flat_with_z(2, 35, "y", 9)])
                .await
                .unwrap();
            assert_no_diff(&lake, "add column", &official, &ours);
            assert_field_ids_are_column_ids(&lake, &ours);
        }
    }
}

/// `set_columns` publishing a table is DDL: the snapshot takes a new schema version, so a
/// DuckDB that attached the lake before sees the table.
#[tokio::test(flavor = "multi_thread")]
async fn a_duckdb_attached_before_sees_the_table_set_columns_publishes() {
    let lake = Lake::new(Backend::Sqlite, 0, BASE).await;
    let ours = lake.copy("ours");
    let attached = lake.duckdb(&ours);
    assert_eq!(
        texts(
            &attached,
            "SELECT table_name FROM duckdb_tables() WHERE database_name = 'lake' ORDER BY 1"
        ),
        ["other"]
    );
    let writer = lake.writer(&ours).await;
    let snapshot = writer.create_snapshot().unwrap();
    let (schema_id, _) = writer.get_or_create_schema("main", None, snapshot).unwrap();
    let (table_id, _) = writer
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    writer
        .set_columns(
            table_id,
            &[ColumnDef::new("id", "int64", true).unwrap()],
            snapshot,
        )
        .unwrap();
    assert_eq!(
        texts(
            &attached,
            "SELECT table_name FROM duckdb_tables() WHERE database_name = 'lake' ORDER BY 1"
        ),
        ["other", "t"]
    );
}

/// A table whose decimal a release before #27 spelled `decimal(2, 1)` takes an append of the
/// same type: the column keeps its row.
#[tokio::test(flavor = "multi_thread")]
async fn an_old_decimal_spelling_takes_an_append() {
    let lake = Lake::new(
        Backend::Sqlite,
        0,
        &format!("{BASE} CREATE TABLE lake.t AS SELECT {FLAT_CTAS};"),
    )
    .await;
    let ours = lake.copy("ours");
    {
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD sqlite; ATTACH '{}' AS c (TYPE sqlite);
                 UPDATE c.ducklake_column SET column_type = 'decimal(2, 1)' WHERE column_name = 'd';",
                ours.display()
            ))
            .unwrap();
    }
    lake.table_writer(lake.writer(&ours).await)
        .append_table("main", "t", &[flat(2, 35, "y")])
        .await
        .unwrap();
    assert_eq!(
        rows(
            &Lake::catalog_database(&ours),
            "SELECT column_id, column_type, begin_snapshot FROM c.ducklake_column
             WHERE column_name = 'd'"
        ),
        ["2|\"decimal(2, 1)\"|3"]
    );
    assert_eq!(
        rows(
            &lake.duckdb(&ours),
            "SELECT count(*), sum(d)::VARCHAR FROM lake.t"
        ),
        ["2|\"6.0\""]
    );
}

/// A table whose columns an earlier release numbered from the catalog-wide counter keeps those
/// ids, and a column added to it takes the table's next id. Its files' field ids stay its
/// column ids, and DuckDB reads every column of every file by them.
#[tokio::test(flavor = "multi_thread")]
async fn a_table_numbered_by_an_earlier_release_keeps_its_ids() {
    let lake = Lake::new(Backend::Sqlite, 0, BASE).await;
    let ours = lake.copy("ours");
    let writer = lake.table_writer(lake.writer(&ours).await);
    writer
        .create_table("main", "t", &[flat(1, 25, "x")])
        .await
        .unwrap();
    drop(writer);
    // Renumber `t`'s columns 1..3 to 11..13, its file's field ids with them: what a release
    // before #27 wrote, drawing ids from one counter across tables.
    {
        let file = texts(
            &lake.duckdb(&ours),
            "SELECT data_file FROM ducklake_list_files('lake', 't')",
        )
        .remove(0);
        let connection = duckdb::Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "LOAD sqlite; ATTACH '{0}' AS c (TYPE sqlite);
                 UPDATE c.ducklake_column SET column_id = column_id + 10,
                                             column_order = column_order + 10
                  WHERE table_id = 2;
                 UPDATE c.ducklake_file_column_stats SET column_id = column_id + 10
                  WHERE table_id = 2;
                 UPDATE c.ducklake_table_column_stats SET column_id = column_id + 10
                  WHERE table_id = 2;
                 COPY (SELECT * FROM read_parquet('{1}'))
                   TO '{1}.renumbered' (FORMAT parquet,
                       FIELD_IDS {{id: 11, d: 12, s: 13}});",
                ours.display(),
                file
            ))
            .unwrap();
        std::fs::rename(format!("{file}.renumbered"), &file).unwrap();
        let size = std::fs::metadata(&file).unwrap().len();
        connection
            .execute_batch(&format!(
                "UPDATE c.ducklake_data_file SET file_size_bytes = {size}, footer_size = NULL
                  WHERE table_id = 2"
            ))
            .unwrap();
    }
    lake.table_writer(lake.writer(&ours).await)
        .append_table("main", "t", &[flat_with_z(2, 35, "y", 9)])
        .await
        .unwrap();
    assert_eq!(
        texts(
            &lake.duckdb(&ours),
            "SELECT column_name || ' ' || column_id
             FROM __ducklake_metadata_lake.ducklake_column WHERE table_id = 2
             ORDER BY column_id"
        ),
        ["id 11", "d 12", "s 13", "z 14"]
    );
    assert_field_ids_are_column_ids(&lake, &ours);
    assert_eq!(
        rows(
            &lake.duckdb(&ours),
            "SELECT id, s, z FROM lake.t ORDER BY id"
        ),
        ["1|\"x\"|NULL", "2|\"y\"|9"]
    );
    lake.assert_ids_distinct(&ours);
}

/// Two writes in flight add a different column each to the same table. Both are offered the
/// table's next id; the one that commits second is refused as a conflict (here by the alter it
/// did not see as well as by the id), and its retry takes the id after.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_new_columns_never_share_an_id() {
    let lake = Lake::new(
        Backend::Sqlite,
        0,
        &format!("{BASE} CREATE TABLE lake.t AS SELECT {FLAT_CTAS};"),
    )
    .await;
    let ours = lake.copy("ours");
    let writer = lake.table_writer(lake.writer(&ours).await);
    let with = |name: &str| {
        let base = flat(2, 35, "y");
        let mut fields = base.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(name, DataType::Int32, true)));
        let mut columns = base.columns().to_vec();
        columns.push(Arc::new(Int32Array::from(vec![7])));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    };
    let (a, b) = (with("a"), with("b"));
    let mut first = writer
        .begin_write(
            "main",
            "t",
            &a.schema(),
            datafusion_ducklake::WriteMode::Append,
        )
        .unwrap();
    let mut second = writer
        .begin_write(
            "main",
            "t",
            &b.schema(),
            datafusion_ducklake::WriteMode::Append,
        )
        .unwrap();
    first.write_batch(&a).unwrap();
    second.write_batch(&b).unwrap();
    first.finish().await.unwrap();
    let refused = second.finish().await.unwrap_err();
    assert!(
        matches!(refused, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "{refused:?}"
    );
    // The retry carries `a` too: an append's batch is the table's columns.
    let retry = {
        let mut fields = a.schema().fields().to_vec();
        fields.push(Arc::new(Field::new("b", DataType::Int32, true)));
        let mut columns = flat(2, 35, "y").columns().to_vec();
        columns.push(Arc::new(Int32Array::from(vec![None::<i32>])));
        columns.push(Arc::new(Int32Array::from(vec![7])));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    };
    writer.append_table("main", "t", &[retry]).await.unwrap();
    lake.assert_ids_distinct(&ours);
    assert_field_ids_are_column_ids(&lake, &ours);
    assert_eq!(
        texts(
            &lake.duckdb(&ours),
            "SELECT column_name || ' ' || column_id
             FROM __ducklake_metadata_lake.ducklake_column WHERE table_id = 2
             ORDER BY column_id"
        ),
        ["id 1", "d 2", "s 3", "a 4", "b 5"]
    );
    assert_eq!(
        rows(
            &lake.duckdb(&ours),
            "SELECT id, a, b FROM lake.t ORDER BY ALL"
        ),
        ["1|NULL|NULL", "2|7|NULL", "2|NULL|7"]
    );
}

/// Two writes in flight make the same new table, each with a different third column. Both
/// are offered ids 1 to 3; the second to commit finds 3 taken by the first's column and is
/// refused as a conflict, where it would otherwise give `b` the id of `a` (which no other check
/// sees: the table's creation is no alter the second write conflicts with).
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_creates_with_different_columns_never_share_an_id() {
    for backend in backends() {
        let lake = Lake::new(backend, 0, BASE).await;
        let ours = lake.copy("ours");
        let writer = lake.table_writer(lake.writer(&ours).await);
        let with = |name: &str, value: i32| {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int64, true),
                    Field::new(name, DataType::Int32, true),
                ])),
                vec![
                    Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                    Arc::new(Int32Array::from(vec![value])),
                ],
            )
            .unwrap()
        };
        let (a, b) = (with("a", 1), with("b", 2));
        let mut first = writer
            .begin_write(
                "main",
                "n",
                &a.schema(),
                datafusion_ducklake::WriteMode::Append,
            )
            .unwrap();
        let mut second = writer
            .begin_write(
                "main",
                "n",
                &b.schema(),
                datafusion_ducklake::WriteMode::Append,
            )
            .unwrap();
        first.write_batch(&a).unwrap();
        second.write_batch(&b).unwrap();
        first.finish().await.unwrap();
        let refused = second.finish().await.unwrap_err();
        assert!(
            matches!(refused, datafusion_ducklake::DuckLakeError::Conflict(_)),
            "{refused:?}"
        );
        lake.assert_ids_distinct(&ours);
        assert_field_ids_are_column_ids(&lake, &ours);
        assert_eq!(rows(&lake.duckdb(&ours), "SELECT * FROM lake.n"), ["1|1"]);
    }
}

/// The DuckDB catalog this crate's DuckDB writer makes numbers each table's columns from 1,
/// spells a decimal and records a column without a default as DuckDB does, gives a table its
/// UUID and a data file its format, and records no insert for an empty `CREATE TABLE`.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn a_duckdb_catalog_numbers_and_types_as_duckdb_does() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let catalog = temp.path().join("catalog.ducklake").display().to_string();
    {
        let metadata = datafusion_ducklake::DuckdbMetadataWriter::new_with_init(&catalog).unwrap();
        metadata.set_data_path(data.to_str().unwrap()).unwrap();
        metadata.create_snapshot().unwrap();
        let metadata: Arc<dyn MetadataWriter> = Arc::new(metadata.with_data_inlining_row_limit(0));
        let writer =
            DuckLakeTableWriter::new(Arc::clone(&metadata), Arc::new(LocalFileSystem::new()))
                .unwrap();
        writer
            .create_table("main", "t", &[flat(1, 25, "x")])
            .await
            .unwrap();
        writer.create_table("main", "n", &[nested()]).await.unwrap();
        writer
            .append_table("main", "t", &[flat_with_z(2, 35, "y", 9)])
            .await
            .unwrap();
        let provider = datafusion_ducklake::DuckdbMetadataProvider::new(&catalog).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog(
            "ducklake",
            Arc::new(DuckLakeCatalog::with_writer(Arc::new(provider), metadata).unwrap()),
        );
        ctx.sql("CREATE TABLE ducklake.main.e (id BIGINT)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    crate::common::ensure_ducklake_installed();
    let connection = duckdb::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(&format!(
            "LOAD ducklake; ATTACH 'ducklake:{catalog}' AS lake (READ_ONLY);"
        ))
        .unwrap();
    assert_eq!(
        rows(
            &connection,
            "SELECT t.table_name, c.column_id, c.column_order, c.column_name, c.column_type,
                    c.default_value, c.default_value_type, c.default_value_dialect
             FROM __ducklake_metadata_lake.ducklake_column c
             JOIN __ducklake_metadata_lake.ducklake_table t USING (table_id)
             ORDER BY t.table_name, c.column_id"
        ),
        [
            r#""e"|1|1|"id"|"int64"|"NULL"|"literal"|"duckdb""#,
            r#""n"|1|1|"n"|"struct"|"NULL"|""|"duckdb""#,
            r#""n"|2|2|"x"|"int32"|"NULL"|"literal"|"duckdb""#,
            r#""n"|3|3|"y"|"list"|"NULL"|""|"duckdb""#,
            r#""n"|4|4|"element"|"varchar"|"NULL"|"literal"|"duckdb""#,
            r#""n"|5|5|"l"|"list"|"NULL"|""|"duckdb""#,
            r#""n"|6|6|"element"|"int32"|"NULL"|"literal"|"duckdb""#,
            r#""t"|1|1|"id"|"int64"|"NULL"|"literal"|"duckdb""#,
            r#""t"|2|2|"d"|"decimal(2,1)"|"NULL"|"literal"|"duckdb""#,
            r#""t"|3|3|"s"|"varchar"|"NULL"|"literal"|"duckdb""#,
            r#""t"|4|4|"z"|"int32"|"NULL"|"literal"|"duckdb""#,
        ]
    );
    assert_eq!(
        rows(
            &connection,
            "SELECT count(*) FILTER (table_uuid IS NULL),
                    (SELECT count(*) FILTER (file_format IS DISTINCT FROM 'parquet')
                     FROM __ducklake_metadata_lake.ducklake_data_file),
                    (SELECT count(*) FROM __ducklake_metadata_lake.ducklake_snapshot_changes
                     WHERE changes_made LIKE '%created_table:\"main\".\"e\"%'
                       AND changes_made LIKE '%inserted_into_table%')
             FROM __ducklake_metadata_lake.ducklake_table"
        ),
        ["0|0|0"]
    );
    assert_eq!(
        rows(&connection, "SELECT id, s, z FROM lake.t ORDER BY id"),
        ["1|\"x\"|NULL", "2|\"y\"|9"]
    );
}

/// Two writes in flight make one new table, `n`, both appending: the first `(id, a)`, the second
/// `second`'s columns. Returns the second's commit, and the lake.
async fn two_creates(
    backend: Backend,
    inlining: usize,
    second: RecordBatch,
) -> (
    Lake,
    PathBuf,
    datafusion_ducklake::Result<datafusion_ducklake::WriteResult>,
) {
    let lake = Lake::new(backend, inlining, BASE).await;
    let ours = lake.copy("ours");
    let writer = lake.table_writer(lake.writer(&ours).await);
    let first = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("a", DataType::Int32, true),
        ])),
        vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef, Arc::new(Int32Array::from(vec![10]))],
    )
    .unwrap();
    let append = datafusion_ducklake::WriteMode::Append;
    let mut one = writer
        .begin_write("main", "n", &first.schema(), append)
        .unwrap();
    let mut two = writer
        .begin_write("main", "n", &second.schema(), append)
        .unwrap();
    one.write_batch(&first).unwrap();
    two.write_batch(&second).unwrap();
    one.finish().await.unwrap();
    let result = two.finish().await;
    drop(writer);
    (lake, ours, result)
}

fn id_and(field: Option<Field>, value: Option<ArrayRef>) -> RecordBatch {
    let mut fields = vec![Field::new("id", DataType::Int64, true)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![2]))];
    fields.extend(field);
    columns.extend(value);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// A second create of the same new table whose `a` is another type is refused: it would put
/// a `varchar` file under the first's `int32` column of the same id, which DuckDB cannot read.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_create_with_another_type_is_refused() {
    for backend in backends() {
        for inlining in [0, 10] {
            let (lake, ours, second) = two_creates(
                backend,
                inlining,
                id_and(
                    Some(Field::new("a", DataType::Utf8, true)),
                    Some(Arc::new(StringArray::from(vec!["hello"]))),
                ),
            )
            .await;
            assert!(
                matches!(second, Err(datafusion_ducklake::DuckLakeError::Conflict(_))),
                "{backend:?} inlining {inlining}: {second:?}"
            );
            assert_eq!(rows(&lake.duckdb(&ours), "SELECT * FROM lake.n"), ["1|10"]);
        }
    }
}

/// A second create of the same new table without `a` is refused: its commit would end the
/// first's `a`, and the first's values with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_create_without_a_column_is_refused() {
    for backend in backends() {
        for inlining in [0, 10] {
            let (lake, ours, second) = two_creates(backend, inlining, id_and(None, None)).await;
            assert!(
                matches!(second, Err(datafusion_ducklake::DuckLakeError::Conflict(_))),
                "{backend:?} inlining {inlining}: {second:?}"
            );
            assert_eq!(rows(&lake.duckdb(&ours), "SELECT * FROM lake.n"), ["1|10"]);
        }
    }
}

/// A second create of the same new table with the same columns appends into it.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_create_with_the_same_columns_appends() {
    for backend in backends() {
        for inlining in [0, 10] {
            let (lake, ours, second) = two_creates(
                backend,
                inlining,
                id_and(
                    Some(Field::new("a", DataType::Int32, true)),
                    Some(Arc::new(Int32Array::from(vec![20]))),
                ),
            )
            .await;
            second.unwrap();
            assert_eq!(
                rows(&lake.duckdb(&ours), "SELECT * FROM lake.n ORDER BY id"),
                ["1|10", "2|20"]
            );
            lake.assert_ids_distinct(&ours);
        }
    }
}
