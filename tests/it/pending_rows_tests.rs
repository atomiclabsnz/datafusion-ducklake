//! Pending catalog rows (`begin_snapshot = PENDING_SNAPSHOT`), against DuckDB 1.5.6 reading the
//! lake (atomiclabsnz/datafusion-ducklake#15).
//!
//! - A write that ends without a commit — an abort, a refused commit, a failed upload, a
//!   session or transaction dropped unfinished — removes the pending schema and table rows its
//!   own start inserted, unless another write's start has taken them up since.
//! - Concurrent starts for one new table commit one table.
//! - No snapshot publishes a table without columns: `get_or_create_table` leaves a new table
//!   pending, and `set_columns` publishes it with its columns. DuckDB loads the lake at every
//!   snapshot after each scenario.
//!
//! The catalogs are official ones (DuckDB made the SQLite one; this crate's DuckDB schema
//! carries the same columns): a pending row's owner is marked by its `schema_uuid` /
//! `table_uuid`.

#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    ColumnDef, DuckLakeError, DuckLakeTableWriter, MetadataWriter, SqliteMetadataWriter,
    TableWriteSession, WriteMode,
};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use tempfile::TempDir;

const PENDING: &str = "9223372036854775807";

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ])
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<String> = ids.iter().map(|id| format!("row {id}")).collect();
    RecordBatch::try_new(
        Arc::new(schema()),
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

fn columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("id", "int32", true).unwrap(),
        ColumnDef::new("name", "varchar", true).unwrap(),
    ]
}

/// A lake: its catalog as DuckDB attaches it (`sqlite:<file>` or a DuckDB file) and its data.
struct Lake {
    _temp: TempDir,
    catalog: String,
    /// DuckDB's `ATTACH` options.
    options: String,
}

impl Lake {
    /// A SQLite catalog DuckDB made.
    fn sqlite() -> Self {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let lake = Self {
            catalog: format!("sqlite:{}", temp.path().join("catalog.sqlite").display()),
            options: format!("DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0", data.display()),
            _temp: temp,
        };
        drop(lake.duckdb(None));
        lake
    }

    /// A DuckDB catalog this crate made.
    #[cfg(feature = "write-duckdb")]
    fn duckdb_catalog() -> (Self, Arc<datafusion_ducklake::DuckdbMetadataWriter>) {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let catalog = temp.path().join("catalog.ducklake").display().to_string();
        let writer = datafusion_ducklake::DuckdbMetadataWriter::new_with_init(&catalog).unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        (
            Self {
                catalog,
                options: "READ_ONLY".to_string(),
                _temp: temp,
            },
            Arc::new(writer.with_data_inlining_row_limit(0)),
        )
    }

    /// The SQLite catalog's writer, rows inlined up to `inlining`.
    async fn sqlite_writer(&self, inlining: usize) -> Arc<SqliteMetadataWriter> {
        let url = format!("{}?mode=rw", self.catalog);
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        Arc::new(writer.with_data_inlining_row_limit(inlining))
    }

    /// DuckDB over the lake, at `snapshot` (the latest when `None`).
    fn duckdb(&self, snapshot: Option<i64>) -> duckdb::Connection {
        crate::common::ensure_ducklake_installed();
        crate::common::ensure_extension_installed("sqlite");
        let connection = duckdb::Connection::open_in_memory().unwrap();
        let at = snapshot
            .map(|snapshot| format!(", SNAPSHOT_VERSION {snapshot}"))
            .unwrap_or_default();
        connection
            .execute_batch(&format!(
                "LOAD ducklake; LOAD sqlite;
                 ATTACH 'ducklake:{}' AS lake ({}{at});",
                self.catalog, self.options
            ))
            .unwrap_or_else(|e| panic!("DuckDB attaches the lake at {snapshot:?}: {e}"));
        connection
    }

    /// The pending schema and table rows, `kind name` each.
    fn pending_rows(&self) -> Vec<String> {
        rows(
            &self.duckdb(None),
            &format!(
                "SELECT 'schema ' || schema_name FROM __ducklake_metadata_lake.ducklake_schema
                  WHERE begin_snapshot = {PENDING}
                 UNION ALL
                 SELECT 'table ' || table_name FROM __ducklake_metadata_lake.ducklake_table
                  WHERE begin_snapshot = {PENDING}
                 ORDER BY 1"
            ),
        )
    }

    /// DuckDB loads the lake at every snapshot, and reads every table it lists there: one line
    /// per snapshot, `snapshot: schema.table=rows …`.
    fn every_snapshot(&self) -> Vec<String> {
        let snapshots: Vec<i64> = rows(
            &self.duckdb(None),
            "SELECT snapshot_id FROM __ducklake_metadata_lake.ducklake_snapshot ORDER BY 1",
        )
        .iter()
        .map(|id| id.parse().unwrap())
        .collect();
        snapshots
            .into_iter()
            .map(|snapshot| {
                let lake = self.duckdb(Some(snapshot));
                let tables = rows(
                    &lake,
                    "SELECT schema_name || '.' || table_name FROM duckdb_tables()
                      WHERE database_name = 'lake' ORDER BY 1",
                );
                let read: Vec<String> = tables
                    .iter()
                    .map(|table| {
                        let (schema, name) = table.split_once('.').unwrap();
                        let count = rows(
                            &lake,
                            &format!("SELECT count(*) FROM lake.\"{schema}\".\"{name}\""),
                        );
                        format!("{table}={}", count[0])
                    })
                    .collect();
                format!("{snapshot}: {}", read.join(" "))
            })
            .collect()
    }
}

/// DuckDB's answer to `sql`, one line per row.
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
                duckdb::types::ValueRef::Text(text) => String::from_utf8_lossy(text).into_owned(),
                duckdb::types::ValueRef::Int(value) => value.to_string(),
                duckdb::types::ValueRef::BigInt(value) => value.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        lines.push(cells.join("|"));
    }
    lines
}

fn table_writer(metadata: Arc<dyn MetadataWriter>) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new())).unwrap()
}

/// Finishes concurrently started sessions, each writing one row, in turn. The first commit
/// makes the table; each later one finds columns another start created with other field ids
/// and is refused `Conflict`, and its retry appends. Returns how many were refused.
async fn finish_in_turn(writer: &DuckLakeTableWriter, sessions: Vec<TableWriteSession>) -> usize {
    let mut refused = 0;
    for (i, mut session) in sessions.into_iter().enumerate() {
        session.write_batch(&batch(&[i as i32])).unwrap();
        match session.finish().await {
            Ok(_) => assert_eq!(refused, 0, "only the first commit makes the table"),
            Err(DuckLakeError::Conflict(_)) => {
                refused += 1;
                writer
                    .append_table("fresh", "t", &[batch(&[i as i32])])
                    .await
                    .unwrap();
            },
            Err(error) => panic!("{error}"),
        }
    }
    refused
}

/// The last line of [`Lake::every_snapshot`].
fn latest(lake: &Lake) -> String {
    lake.every_snapshot().pop().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_aborted_write_leaves_no_pending_rows() {
    for inlining in [10, 0] {
        let lake = Lake::sqlite();
        let writer = table_writer(lake.sqlite_writer(inlining).await);
        let mut session = writer
            .begin_write("fresh", "t", &schema(), WriteMode::Create)
            .unwrap();
        session.write_batch(&batch(&[1])).unwrap();
        assert_eq!(lake.pending_rows(), ["schema fresh", "table t"]);
        session.abort().await.unwrap();
        assert_eq!(
            lake.pending_rows(),
            Vec::<String>::new(),
            "inlining {inlining}"
        );
        assert_eq!(lake.every_snapshot(), ["0: "]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_session_or_transaction_leaves_no_pending_rows() {
    let lake = Lake::sqlite();
    let writer = table_writer(lake.sqlite_writer(0).await);
    {
        let mut session = writer
            .begin_write("fresh", "t", &schema(), WriteMode::Append)
            .unwrap();
        session.write_batch(&batch(&[1])).unwrap();
    }
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
    {
        let mut transaction = writer.transaction();
        transaction
            .stage_write("fresh", "u", &schema(), WriteMode::Append, &[batch(&[1])])
            .await
            .unwrap();
        assert_eq!(lake.pending_rows(), ["schema fresh", "table u"]);
        transaction.abort().await.unwrap();
    }
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
    assert_eq!(lake.every_snapshot(), ["0: "]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_commit_leaves_no_pending_rows() {
    // A view of the table's name, made after the write began, refuses its commit.
    for inlining in [10, 0] {
        let lake = Lake::sqlite();
        let writer = table_writer(lake.sqlite_writer(inlining).await);
        let mut session = writer
            .begin_write("main", "t", &schema(), WriteMode::Create)
            .unwrap();
        session.write_batch(&batch(&[1])).unwrap();
        lake.duckdb(None)
            .execute_batch("CREATE VIEW lake.main.t AS SELECT 1 AS id")
            .unwrap();
        assert!(matches!(
            session.finish().await.unwrap_err(),
            DuckLakeError::ViewExists(_)
        ));
        assert_eq!(
            lake.pending_rows(),
            Vec::<String>::new(),
            "inlining {inlining}"
        );
        assert_eq!(lake.every_snapshot(), ["0: ", "1: "]);
    }
}

/// The local filesystem, refusing every upload.
#[derive(Debug)]
struct RefusingStore(LocalFileSystem);

impl std::fmt::Display for RefusingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RefusingStore")
    }
}

fn refused() -> object_store::Error {
    object_store::Error::Generic {
        store: "RefusingStore",
        source: "upload refused".into(),
    }
}

#[async_trait::async_trait]
impl ObjectStore for RefusingStore {
    async fn put_opts(
        &self,
        _location: &ObjectPath,
        _payload: object_store::PutPayload,
        _opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        Err(refused())
    }

    async fn put_multipart_opts(
        &self,
        _location: &ObjectPath,
        _opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        Err(refused())
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.0.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectPath>> {
        self.0.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.0.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.0.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.0.copy_opts(from, to, options).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_upload_leaves_no_pending_rows() {
    let lake = Lake::sqlite();
    let metadata: Arc<dyn MetadataWriter> = lake.sqlite_writer(0).await;
    let writer =
        DuckLakeTableWriter::new(metadata, Arc::new(RefusingStore(LocalFileSystem::new())))
            .unwrap();
    writer
        .create_table("fresh", "t", &[batch(&[1, 2])])
        .await
        .unwrap_err();
    let mut session = writer
        .begin_write("fresh", "u", &schema(), WriteMode::Create)
        .unwrap();
    session.write_batch(&batch(&[1])).unwrap();
    session.finish().await.unwrap_err();
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
    assert_eq!(lake.every_snapshot(), ["0: "]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_row_another_write_took_up_is_left_to_it() {
    // Two writes into one new table: the first's start makes its rows, the second's takes them
    // up. The first aborting leaves them to the second, which commits into them.
    let lake = Lake::sqlite();
    let writer = table_writer(lake.sqlite_writer(0).await);
    let mut first = writer
        .begin_write("fresh", "t", &schema(), WriteMode::Append)
        .unwrap();
    first.write_batch(&batch(&[1])).unwrap();
    let mut second = writer
        .begin_write("fresh", "t", &schema(), WriteMode::Append)
        .unwrap();
    second.write_batch(&batch(&[2])).unwrap();
    first.abort().await.unwrap();
    assert_eq!(lake.pending_rows(), ["schema fresh", "table t"]);
    second.finish().await.unwrap();
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
    assert_eq!(lake.every_snapshot(), ["0: ", "1: fresh.t=1"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_starts_for_a_new_table_commit_one_table() {
    // Writers on separate connections, as separate processes would be, start at once.
    let lake = Lake::sqlite();
    let mut writers = Vec::new();
    for _ in 0..4 {
        writers.push(table_writer(lake.sqlite_writer(0).await));
    }
    let barrier = Arc::new(std::sync::Barrier::new(writers.len()));
    let sessions: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = writers
            .iter()
            .map(|writer| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    // A busy catalog refuses a start; the caller retries it.
                    loop {
                        match writer.begin_write("fresh", "t", &schema(), WriteMode::Append) {
                            Ok(session) => return session,
                            Err(error) => assert!(
                                error.to_string().contains("locked")
                                    || error.to_string().contains("busy"),
                                "{error}"
                            ),
                        }
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(finish_in_turn(&writers[0], sessions).await, 3);
    assert_eq!(
        rows(
            &lake.duckdb(None),
            "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_table WHERE table_name = 't'"
        ),
        ["1"],
        "one table row"
    );
    assert_eq!(latest(&lake), "4: fresh.t=4");
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn get_or_create_table_publishes_no_table_without_columns() {
    let lake = Lake::sqlite();
    let metadata = lake.sqlite_writer(0).await;
    let snapshot = metadata.create_snapshot().unwrap();
    let (schema_id, _) = metadata
        .get_or_create_schema("main", None, snapshot)
        .unwrap();
    let (table_id, created) = metadata
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    assert!(created);
    // Between the two calls, DuckDB loads the lake, and the table is not in it yet.
    assert_eq!(lake.every_snapshot(), ["0: ", "1: "]);
    metadata
        .set_columns(table_id, &columns(), snapshot)
        .unwrap();
    assert_eq!(lake.every_snapshot(), ["0: ", "1: main.t=0"]);
    assert_eq!(
        rows(
            &lake.duckdb(None),
            "SELECT changes_made FROM __ducklake_metadata_lake.ducklake_snapshot_changes
              WHERE snapshot_id = 1"
        ),
        ["created_table:\"main\".\"t\""]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pending_table_get_or_create_takes_up_is_published_only_with_its_columns() {
    // A write in flight has made `t`; `get_or_create_table` takes it up, and `set_columns`
    // publishes it. At no snapshot is it a table without columns.
    let lake = Lake::sqlite();
    let metadata = lake.sqlite_writer(0).await;
    let writer = table_writer(Arc::clone(&metadata) as Arc<dyn MetadataWriter>);
    let mut late = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    late.write_batch(&batch(&[1])).unwrap();
    let snapshot = metadata.create_snapshot().unwrap();
    let (schema_id, _) = metadata
        .get_or_create_schema("main", None, snapshot)
        .unwrap();
    let (table_id, created) = metadata
        .get_or_create_table(schema_id, "t", None, snapshot)
        .unwrap();
    assert!(created);
    assert_eq!(lake.every_snapshot(), ["0: ", "1: "]);
    metadata
        .set_columns(table_id, &columns(), snapshot)
        .unwrap();
    // `set_columns` gave the columns ids of its own: the write in flight is refused, as a
    // concurrent create is, and its retry appends.
    assert!(matches!(
        late.finish().await.unwrap_err(),
        DuckLakeError::Conflict(_)
    ));
    writer
        .append_table("main", "t", &[batch(&[1])])
        .await
        .unwrap();
    assert_eq!(lake.every_snapshot(), ["0: ", "1: main.t=0", "2: main.t=1"]);
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread")]
async fn get_or_create_naming_another_path_takes_a_row_up_where_it_is() {
    // A write in flight into a new schema and table puts its files under their rows' paths.
    // `get_or_create_*` naming other paths takes the rows up as they are; the write commits and
    // every row is where the catalog says (#29's rule: no take-up re-paths a row).
    let lake = Lake::sqlite();
    let metadata = lake.sqlite_writer(0).await;
    let writer = table_writer(Arc::clone(&metadata) as Arc<dyn MetadataWriter>);
    let mut late = writer
        .begin_write("fresh", "t", &schema(), WriteMode::Append)
        .unwrap();
    late.write_batch(&batch(&[1, 2])).unwrap();
    let snapshot = metadata.create_snapshot().unwrap();
    let (schema_id, created) = metadata
        .get_or_create_schema("fresh", Some("elsewhere"), snapshot)
        .unwrap();
    assert!(created);
    let (_, created) = metadata
        .get_or_create_table(schema_id, "t", Some("other"), snapshot)
        .unwrap();
    assert!(created);
    late.finish().await.unwrap();
    assert_eq!(lake.every_snapshot(), ["0: ", "1: ", "2: fresh.t=2"]);
    assert_eq!(
        rows(
            &lake.duckdb(None),
            "SELECT id FROM lake.fresh.t ORDER BY id"
        ),
        ["1", "2"]
    );
    assert_eq!(
        rows(
            &lake.duckdb(None),
            "SELECT s.path || t.path FROM __ducklake_metadata_lake.ducklake_table t
               JOIN __ducklake_metadata_lake.ducklake_schema s USING (schema_id)
              WHERE t.table_name = 't'"
        ),
        ["fresh/t/"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_aborted_transaction_releases_rows_its_own_stages_took_up() {
    // Two stages into one new table, and two into two new tables of one new schema: the later
    // stages take up the earlier ones' rows, and an abort releases all of them.
    let lake = Lake::sqlite();
    let writer = table_writer(lake.sqlite_writer(0).await);
    let mut transaction = writer.transaction();
    for table in ["t", "t", "u"] {
        transaction
            .stage_write("fresh", table, &schema(), WriteMode::Append, &[batch(&[1])])
            .await
            .unwrap();
    }
    assert_eq!(lake.pending_rows(), ["schema fresh", "table t", "table u"]);
    transaction.abort().await.unwrap();
    assert_eq!(lake.pending_rows(), Vec::<String>::new());
    assert_eq!(lake.every_snapshot(), ["0: "]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_leaves_a_row_another_write_took_up_between_its_stages() {
    let lake = Lake::sqlite();
    let writer = table_writer(lake.sqlite_writer(0).await);
    let mut transaction = writer.transaction();
    transaction
        .stage_write("fresh", "t", &schema(), WriteMode::Append, &[batch(&[1])])
        .await
        .unwrap();
    let mut other = writer
        .begin_write("fresh", "t", &schema(), WriteMode::Append)
        .unwrap();
    other.write_batch(&batch(&[2])).unwrap();
    transaction
        .stage_write("fresh", "t", &schema(), WriteMode::Append, &[batch(&[3])])
        .await
        .unwrap();
    transaction.abort().await.unwrap();
    assert_eq!(lake.pending_rows(), ["schema fresh", "table t"]);
    other.finish().await.unwrap();
    assert_eq!(lake.every_snapshot(), ["0: ", "1: fresh.t=1"]);
}

/// The DuckDB catalog: one connection per writer, so its starts are serialized; and the same
/// release and publication rules. DuckDB opens a catalog file for one client, so the oracle
/// reads it once the writer is dropped.
#[cfg(feature = "write-duckdb")]
mod duckdb_catalog {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn an_aborted_write_leaves_no_pending_rows_and_a_taken_row_is_left() {
        let (lake, metadata) = Lake::duckdb_catalog();
        {
            let writer = table_writer(metadata);
            let mut aborted = writer
                .begin_write("fresh", "gone", &schema(), WriteMode::Create)
                .unwrap();
            aborted.write_batch(&batch(&[1])).unwrap();
            aborted.abort().await.unwrap();

            let mut first = writer
                .begin_write("fresh", "t", &schema(), WriteMode::Append)
                .unwrap();
            first.write_batch(&batch(&[1])).unwrap();
            let mut second = writer
                .begin_write("fresh", "t", &schema(), WriteMode::Append)
                .unwrap();
            second.write_batch(&batch(&[2])).unwrap();
            first.abort().await.unwrap();
            second.finish().await.unwrap();
        }
        assert_eq!(lake.pending_rows(), Vec::<String>::new());
        assert_eq!(lake.every_snapshot(), ["1: fresh.t=1"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_starts_for_a_new_table_commit_one_table() {
        let (lake, metadata) = Lake::duckdb_catalog();
        {
            let writer = table_writer(metadata);
            let barrier = Arc::new(std::sync::Barrier::new(4));
            let sessions: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let writer = &writer;
                        scope.spawn(move || {
                            barrier.wait();
                            writer
                                .begin_write("fresh", "t", &schema(), WriteMode::Append)
                                .unwrap()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            assert_eq!(finish_in_turn(&writer, sessions).await, 3);
        }
        assert_eq!(
            rows(
                &lake.duckdb(None),
                "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_table
                  WHERE table_name = 't'"
            ),
            ["1"]
        );
        assert_eq!(latest(&lake), "4: fresh.t=4");
        assert_eq!(lake.pending_rows(), Vec::<String>::new());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn starts_on_two_writers_of_one_catalog_file_make_one_table() {
        // Two writers opened on one file, each its own connection, start at once. (DuckDB
        // lets one process open a catalog file twice.)
        let (lake, first) = Lake::duckdb_catalog();
        let second =
            Arc::new(datafusion_ducklake::DuckdbMetadataWriter::new(&lake.catalog).unwrap());
        let results: Vec<String> = {
            let writers = [table_writer(first), table_writer(second)];
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let started: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = writers
                    .iter()
                    .map(|writer| {
                        let barrier = Arc::clone(&barrier);
                        scope.spawn(move || {
                            barrier.wait();
                            writer.begin_write("fresh", "t", &schema(), WriteMode::Append)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut results = Vec::new();
            for (i, started) in started.into_iter().enumerate() {
                results.push(match started {
                    Ok(mut session) => {
                        session.write_batch(&batch(&[i as i32])).unwrap();
                        match session.finish().await {
                            Ok(_) => "committed".to_string(),
                            Err(error) => format!("finish: {error:?}"),
                        }
                    },
                    Err(error) => format!("start: {error:?}"),
                });
            }
            results
        };
        // Both commit: the second start takes up the first's row, or finds its table.
        assert_eq!(results, ["committed", "committed"]);
        assert_eq!(lake.pending_rows(), Vec::<String>::new());
        assert_eq!(
            rows(
                &lake.duckdb(None),
                "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_table
                  WHERE table_name = 't' AND end_snapshot IS NULL"
            ),
            ["1"]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_or_create_table_publishes_no_table_without_columns() {
        let (lake, metadata) = Lake::duckdb_catalog();
        let table_id;
        {
            let snapshot = metadata.create_snapshot().unwrap();
            let (schema_id, _) = metadata
                .get_or_create_schema("main", None, snapshot)
                .unwrap();
            table_id = metadata
                .get_or_create_table(schema_id, "t", None, snapshot)
                .unwrap()
                .0;
        }
        drop(metadata);
        assert_eq!(lake.every_snapshot(), ["1: "]);
        let metadata = datafusion_ducklake::DuckdbMetadataWriter::new(&lake.catalog).unwrap();
        metadata.set_columns(table_id, &columns(), 1).unwrap();
        drop(metadata);
        assert_eq!(lake.every_snapshot(), ["1: main.t=0"]);
    }
}
