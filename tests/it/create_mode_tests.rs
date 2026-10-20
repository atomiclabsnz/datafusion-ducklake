#![cfg(all(feature = "write-sqlite", feature = "write"))]
//! `WriteMode::Create`, the write of `CREATE TABLE … AS`: a table that exists is refused, and
//! so is one another writer creates after this write began, inside the commit's own
//! transaction. A live view with the table's name is refused (`ViewExists`), in every mode
//! that would make the table. Each case runs
//! with the rows inlined in the catalog and with them in Parquet, the two ways a write commits.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, SqliteMetadataProvider,
    SqliteMetadataWriter, WriteMode,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

/// A table writer over a fresh SQLite catalog; `inlining` 0 puts every row in Parquet.
async fn table_writer(temp: &TempDir, inlining: usize) -> DuckLakeTableWriter {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.db").display()
    );
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer.with_data_inlining_row_limit(inlining));
    DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
}

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ])
}

fn batch(id: i32, name: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(schema()),
        vec![Arc::new(Int32Array::from(vec![id])), Arc::new(StringArray::from(vec![name]))],
    )
    .unwrap()
}

/// `main.t`'s names, read back through a catalog over the same database.
async fn names(temp: &TempDir) -> Vec<String> {
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let batches = ctx
        .sql("SELECT name FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|b| {
            // As text, whatever string type the scan returns (`Utf8View` in DataFusion 55).
            let names = b.column(0);
            (0..names.len())
                .map(|i| arrow::util::display::array_value_to_string(names, i).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Every file under the lake's data directory.
fn files(temp: &TempDir) -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
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
    walk(&temp.path().join("data"), &mut found);
    found.sort();
    found
}

fn assert_exists(error: DuckLakeError) {
    assert!(
        matches!(error, DuckLakeError::TableExists(_)),
        "a create of a table that exists is TableExists: {error:?}"
    );
}

fn assert_view_exists(error: DuckLakeError) {
    assert!(
        matches!(error, DuckLakeError::ViewExists(_)),
        "a table made where a view has its name is ViewExists: {error:?}"
    );
}

/// Adds a live view named `{name}` to the schema of table `existing`, as another writer
/// (official DuckLake) would. The same statement on every backend, so its columns go by
/// position (`view_id, view_uuid, begin_snapshot, end_snapshot, schema_id, view_name,
/// dialect, sql, column_aliases`): MySQL reserves `sql` and quotes names its own way.
const ADD_VIEW: &str = "INSERT INTO ducklake_view
     SELECT 1000, '00000000-0000-0000-0000-000000000001', MAX(snapshot_id), NULL,
            (SELECT schema_id FROM ducklake_table
             WHERE table_name = 'existing' AND end_snapshot IS NULL),
            '{name}', 'duckdb', 'SELECT 1 AS id', ''
     FROM ducklake_snapshot";

fn add_view_sql(name: &str) -> String {
    ADD_VIEW.replace("{name}", name)
}

async fn add_sqlite_view(temp: &TempDir, name: &str) {
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(add_view_sql(name)))
        .execute(&pool)
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_create_of_a_table_that_exists_is_refused_and_writes_nothing() {
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        writer
            .write_table("main", "t", &[batch(1, "kept")])
            .await
            .unwrap();
        let files_before = files(&temp);
        assert_exists(
            writer
                .create_table("main", "t", &[batch(2, "created")])
                .await
                .unwrap_err(),
        );
        assert_eq!(names(&temp).await, ["kept"], "inlining {inlining}");
        // A refused create leaves no Parquet of its own behind.
        assert_eq!(files(&temp), files_before, "inlining {inlining}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_table_another_writer_creates_first_refuses_the_create_at_commit() {
    // The race CREATE TABLE … AS has between finding a table absent and committing it: a
    // second writer creates the same table in between. Its rows are its own.
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        let mut late = writer
            .begin_write("main", "t", &schema(), WriteMode::Create)
            .unwrap();
        late.write_batch(&batch(9, "late")).unwrap();
        writer
            .create_table("main", "t", &[batch(1, "first")])
            .await
            .unwrap();
        assert_exists(late.finish().await.unwrap_err());
        assert_eq!(names(&temp).await, ["first"], "inlining {inlining}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_view_another_writer_creates_first_refuses_the_create_at_commit() {
    // The same race, with a view of the table's name: the commit refuses it, and the name
    // stays the view's.
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        writer
            .write_table("main", "existing", &[batch(1, "kept")])
            .await
            .unwrap();
        let mut late = writer
            .begin_write("main", "t", &schema(), WriteMode::Create)
            .unwrap();
        late.write_batch(&batch(9, "late")).unwrap();
        add_sqlite_view(&temp, "t").await;
        assert_view_exists(late.finish().await.unwrap_err());

        let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
        let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        let main = datafusion::catalog::CatalogProvider::schema(&catalog, "main").unwrap();
        assert_eq!(
            main.table_type("t").await.unwrap(),
            Some(datafusion::logical_expr::TableType::View),
            "inlining {inlining}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_replace_or_append_that_would_make_a_table_beside_a_view_is_refused_at_commit() {
    // `CREATE OR REPLACE TABLE` writes Replace, and an INSERT into a table it found absent
    // Append: neither makes a table where a view created since has the name.
    for mode in [WriteMode::Replace, WriteMode::Append] {
        for inlining in [10, 0] {
            let temp = TempDir::new().unwrap();
            let writer = table_writer(&temp, inlining).await;
            writer
                .write_table("main", "existing", &[batch(1, "kept")])
                .await
                .unwrap();
            let mut late = writer.begin_write("main", "t", &schema(), mode).unwrap();
            late.write_batch(&batch(9, "late")).unwrap();
            add_sqlite_view(&temp, "t").await;
            assert_view_exists(late.finish().await.unwrap_err());
            writer
                .create_table("main", "next", &[batch(2, "next")])
                .await
                .unwrap();
            assert_eq!(
                column_less_tables(&temp).await,
                0,
                "{mode:?} inlining {inlining}"
            );

            let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
            let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
            let catalog = DuckLakeCatalog::new(provider).unwrap();
            let main = datafusion::catalog::CatalogProvider::schema(&catalog, "main").unwrap();
            assert_eq!(
                main.table_type("t").await.unwrap(),
                Some(datafusion::logical_expr::TableType::View),
                "{mode:?} inlining {inlining}: the name stays the view's"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_replace_of_a_table_that_exists_is_not_refused_as_a_view() {
    // The view check is the write that makes the table's: replacing and appending into a
    // table that exists are not asked it.
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        writer
            .write_table("main", "t", &[batch(1, "a")])
            .await
            .unwrap();
        writer
            .write_table("main", "t", &[batch(2, "b")])
            .await
            .unwrap();
        writer
            .append_table("main", "t", &[batch(3, "c")])
            .await
            .unwrap();
        assert_eq!(names(&temp).await, ["b", "c"], "inlining {inlining}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn register_table_over_a_view_is_refused_as_a_typed_error() {
    let temp = TempDir::new().unwrap();
    let writer = table_writer(&temp, 10).await;
    writer
        .write_table("main", "existing", &[batch(1, "kept")])
        .await
        .unwrap();
    add_sqlite_view(&temp, "t").await;

    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let metadata = SqliteMetadataWriter::new(&format!("{conn}?mode=rw"))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(metadata)).unwrap();
    let main = datafusion::catalog::CatalogProvider::schema(&catalog, "main").unwrap();
    let empty =
        datafusion::datasource::MemTable::try_new(Arc::new(schema()), vec![vec![]]).unwrap();
    let error = main
        .register_table("t".to_string(), Arc::new(empty))
        .expect_err("a view's name is not registered over");
    match error {
        datafusion::error::DataFusionError::External(error) => assert!(
            matches!(
                error.downcast_ref::<DuckLakeError>(),
                Some(DuckLakeError::ViewExists(_))
            ),
            "{error}"
        ),
        error => panic!("a view's name is refused as ViewExists: {error:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_create_of_a_new_table_with_rows_commits_once() {
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        let created = writer
            .create_table("main", "t", &[batch(1, "a")])
            .await
            .unwrap();
        writer
            .append_table("main", "t", &[batch(2, "b")])
            .await
            .unwrap();
        assert_eq!(names(&temp).await, ["a", "b"], "inlining {inlining}");
        // Snapshot 0 is the catalog's own; the create is snapshot 1, one commit.
        assert_eq!(created.snapshot_id, 1, "inlining {inlining}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn register_table_never_replaces_a_table() {
    // DataFusion asks `register_table` only of a table it found absent; one another writer
    // created since is refused, not replaced by an empty table of the same columns.
    let temp = TempDir::new().unwrap();
    let writer = table_writer(&temp, 10).await;
    writer
        .write_table("main", "t", &[batch(1, "kept")])
        .await
        .unwrap();

    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
    let metadata = SqliteMetadataWriter::new(&format!("{conn}?mode=rw"))
        .await
        .unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(metadata)).unwrap();
    let main = datafusion::catalog::CatalogProvider::schema(&catalog, "main").unwrap();
    let empty =
        datafusion::datasource::MemTable::try_new(Arc::new(schema()), vec![vec![]]).unwrap();
    let error = main
        .register_table("t".to_string(), Arc::new(empty))
        .expect_err("a table that exists is not registered over");
    assert!(
        error.to_string().contains("Table already exists"),
        "{error}"
    );
    assert_eq!(names(&temp).await, ["kept"]);
}

/// The Create cases on a backend's own writer: a table that exists, a table another writer
/// creates while this one is writing, and a new table. The commit's check is each backend's
/// own code, so each runs it. `add_view`, when given, adds a view `viewed` ([`ADD_VIEW`]) on
/// the backend, as another writer would while this one is writing.
async fn create_refusals(
    metadata: Arc<dyn MetadataWriter>,
    add_view: Option<impl std::future::Future<Output = ()>>,
) {
    let writer = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new())).unwrap();
    writer
        .write_table("main", "existing", &[batch(1, "kept")])
        .await
        .unwrap();
    assert_exists(
        writer
            .create_table("main", "existing", &[batch(2, "created")])
            .await
            .unwrap_err(),
    );

    let mut late = writer
        .begin_write("main", "raced", &schema(), WriteMode::Create)
        .unwrap();
    late.write_batch(&batch(9, "late")).unwrap();
    writer
        .create_table("main", "raced", &[batch(1, "first")])
        .await
        .unwrap();
    assert_exists(late.finish().await.unwrap_err());

    writer
        .create_table("main", "fresh", &[batch(1, "a")])
        .await
        .unwrap();

    // Another table's commit lands mid-write: the new table is still this commit's.
    let mut during = writer
        .begin_write("main", "during", &schema(), WriteMode::Create)
        .unwrap();
    during.write_batch(&batch(1, "a")).unwrap();
    writer
        .create_table("main", "between", &[batch(2, "b")])
        .await
        .unwrap();
    during.finish().await.unwrap();

    if let Some(add_view) = add_view {
        // A Create, and a Replace or Append that would make the table, all begun before
        // the view: each commit refuses it.
        let writes = [WriteMode::Create, WriteMode::Replace, WriteMode::Append].map(|mode| {
            let mut late = writer
                .begin_write("main", "viewed", &schema(), mode)
                .unwrap();
            late.write_batch(&batch(9, "late")).unwrap();
            late
        });
        add_view.await;
        for late in writes {
            assert_view_exists(late.finish().await.unwrap_err());
        }
    }
}

/// Where a test container's mapped ports answer: `TESTCONTAINERS_HOST_OVERRIDE` when set (a
/// devcontainer whose Docker is a sibling service; `get_host` names the API endpoint
/// there, which takes no database connections), else what testcontainers says.
#[cfg(any(feature = "write-mysql", feature = "write-postgres"))]
async fn container_host<I: testcontainers::Image>(
    container: &testcontainers::ContainerAsync<I>,
) -> String {
    match std::env::var("TESTCONTAINERS_HOST_OVERRIDE") {
        Ok(host) if !host.is_empty() => host,
        _ => container.get_host().await.unwrap().to_string(),
    }
}

fn data_path(temp: &TempDir) -> String {
    let path = temp.path().join("data");
    std::fs::create_dir_all(&path).unwrap();
    path.to_str().unwrap().to_string()
}

#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn create_is_refused_on_duckdb_as_on_sqlite() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.duckdb");
    let writer =
        datafusion_ducklake::DuckdbMetadataWriter::new_with_init(path.to_str().unwrap()).unwrap();
    writer.set_data_path(&data_path(&temp)).unwrap();
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
    create_refusals(Arc::clone(&writer), None::<std::future::Ready<()>>).await;
    abandon_a_write_then_commit(writer).await;
    let connection = duckdb::Connection::open(&path).unwrap();
    let column_less: i64 = connection
        .query_row(VISIBLE_COLUMN_LESS, [], |row| row.get(0))
        .unwrap();
    assert_eq!(column_less, 0, "an abandoned write publishes no table");
    drop(connection);

    // DuckDB allows one client per catalog file, so the view is added with the writer
    // closed. Nothing but the commit checks for a view, so this runs that check.
    let connection = duckdb::Connection::open(&path).unwrap();
    connection.execute(&add_view_sql("viewed"), []).unwrap();
    drop(connection);
    let writer = datafusion_ducklake::DuckdbMetadataWriter::new(path.to_str().unwrap()).unwrap();
    let writer =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    assert_view_exists(
        writer
            .create_table("main", "viewed", &[batch(1, "a")])
            .await
            .unwrap_err(),
    );
    assert_view_exists(
        writer
            .write_table("main", "viewed", &[batch(1, "a")])
            .await
            .unwrap_err(),
    );
    assert_view_exists(
        writer
            .append_table("main", "viewed", &[batch(1, "a")])
            .await
            .unwrap_err(),
    );
}

#[cfg(feature = "write-mysql")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn create_is_refused_on_mysql_as_on_sqlite() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::mysql::Mysql::default()
        .start()
        .await
        .unwrap();
    let host = container_host(&container).await;
    let port = container.get_host_port_ipv4(3306).await.unwrap();
    let writer = datafusion_ducklake::MySqlMetadataWriter::new_with_init(&format!(
        "mysql://root@{host}:{port}/test"
    ))
    .await
    .unwrap();
    let temp = TempDir::new().unwrap();
    writer.set_data_path(&data_path(&temp)).unwrap();
    let pool = sqlx::MySqlPool::connect(&format!("mysql://root@{host}:{port}/test"))
        .await
        .unwrap();
    let add_view = async {
        sqlx::query(sqlx::AssertSqlSafe(add_view_sql("viewed")))
            .execute(&pool)
            .await
            .unwrap();
    };
    let writer: Arc<dyn MetadataWriter> = Arc::new(writer);
    create_refusals(Arc::clone(&writer), Some(add_view)).await;
    abandon_a_write_then_commit(writer).await;
    let column_less: i64 = sqlx::query_scalar(VISIBLE_COLUMN_LESS)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(column_less, 0, "an abandoned write publishes no table");
}

#[cfg(feature = "write-postgres")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn create_is_refused_on_both_postgres_writers_as_on_sqlite() {
    use testcontainers::runners::AsyncRunner;
    let container = testcontainers_modules::postgres::Postgres::default()
        .start()
        .await
        .unwrap();
    let host = container_host(&container).await;
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let base = format!("postgresql://postgres:postgres@{host}:{port}");

    // Each writer in a database of its own.
    let admin = sqlx::PgPool::connect(&format!("{base}/postgres"))
        .await
        .unwrap();
    for database in ["single", "multi"] {
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {database}")))
            .execute(&admin)
            .await
            .unwrap();
    }

    let temp = TempDir::new().unwrap();
    let single = datafusion_ducklake::PostgresSingleCatalogMetadataWriter::new_with_init(&format!(
        "{base}/single"
    ))
    .await
    .unwrap();
    single.set_data_path(&data_path(&temp)).unwrap();
    let single_pool = sqlx::PgPool::connect(&format!("{base}/single"))
        .await
        .unwrap();
    let add_view = async {
        sqlx::query(sqlx::AssertSqlSafe(add_view_sql("viewed")))
            .execute(&single_pool)
            .await
            .unwrap();
    };
    let single: Arc<dyn MetadataWriter> = Arc::new(single);
    create_refusals(Arc::clone(&single), Some(add_view)).await;
    abandon_a_write_then_commit(single).await;
    let single_pool = sqlx::PgPool::connect(&format!("{base}/single"))
        .await
        .unwrap();
    let column_less: i64 = sqlx::query_scalar(VISIBLE_COLUMN_LESS)
        .fetch_one(&single_pool)
        .await
        .unwrap();
    assert_eq!(
        column_less, 0,
        "single-catalog: an abandoned write publishes no table"
    );

    let pool = sqlx::PgPool::connect(&format!("{base}/multi"))
        .await
        .unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let catalog_id = datafusion_ducklake::MulticatalogManager::new(pool.clone())
        .create_catalog("lake")
        .await
        .unwrap();
    let multi = datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog_id)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    multi.set_data_path(&data_path(&temp)).unwrap();
    let add_view = async {
        sqlx::query(sqlx::AssertSqlSafe(add_view_sql("viewed")))
            .execute(&pool)
            .await
            .unwrap();
    };
    let multi: Arc<dyn MetadataWriter> = Arc::new(multi);
    create_refusals(Arc::clone(&multi), Some(add_view)).await;
    abandon_a_write_then_commit(multi).await;
    let column_less: i64 = sqlx::query_scalar(VISIBLE_COLUMN_LESS)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        column_less, 0,
        "multi-catalog: an abandoned write publishes no table"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_create_survives_an_unrelated_commit_during_its_write() {
    // Another table's commit lands while this create writes: the new table is still this
    // commit's, with its row.
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        let mut a = writer
            .begin_write("main", "a", &schema(), WriteMode::Create)
            .unwrap();
        a.write_batch(&batch(1, "a")).unwrap();
        writer
            .create_table("main", "b", &[batch(2, "b")])
            .await
            .unwrap();
        a.finish()
            .await
            .unwrap_or_else(|e| panic!("inlining {inlining}: {e:?}"));
        let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
        let provider = SqliteMetadataProvider::new(&conn).await.unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
        let rows = ctx
            .sql("SELECT name FROM lake.main.a")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let count: usize = rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(count, 1, "inlining {inlining}: table a reads back its row");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_create_of_a_table_that_exists_says_so_whatever_its_columns() {
    // Refused as the table existing, not as a type change into it: the write's start
    // refuses a create into a table with live columns before it compares them.
    let temp = TempDir::new().unwrap();
    let writer = table_writer(&temp, 10).await;
    writer
        .write_table("main", "t", &[batch(1, "kept")])
        .await
        .unwrap();
    let other = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(vec!["x"]))],
    )
    .unwrap();
    assert_exists(
        writer
            .create_table("main", "t", &[other])
            .await
            .unwrap_err(),
    );
}

/// [`column_less_tables`]'s count, in SQL every backend takes.
const VISIBLE_COLUMN_LESS: &str = "SELECT count(*) FROM ducklake_table t \
     JOIN ducklake_schema s ON s.schema_id = t.schema_id \
     WHERE t.end_snapshot IS NULL AND s.end_snapshot IS NULL \
     AND t.begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot) \
     AND s.begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot) \
     AND NOT EXISTS (SELECT 1 FROM ducklake_column c \
                     WHERE c.table_id = t.table_id AND c.end_snapshot IS NULL \
                     AND c.begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot))";

/// A write that begins and never commits — a tab closed mid-write — and then another table's
/// commit, which takes the snapshot the abandoned write's start expected to.
async fn abandon_a_write_then_commit(metadata: Arc<dyn MetadataWriter>) {
    let writer = DuckLakeTableWriter::new(metadata, Arc::new(LocalFileSystem::new())).unwrap();
    {
        let mut abandoned = writer
            .begin_write("main", "abandoned", &schema(), WriteMode::Append)
            .unwrap();
        abandoned.write_batch(&batch(1, "lost")).unwrap();
    }
    writer
        .create_table("main", "after", &[batch(2, "after")])
        .await
        .unwrap();
}

/// Tables a reader sees at the current snapshot that have no live columns: what DuckDB
/// refuses to load a lake over ("Table entry ... does not have any columns"). A row no snapshot
/// reaches yet is no reader's, so it is not counted.
async fn column_less_tables(temp: &TempDir) -> i64 {
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    sqlx::query_scalar(VISIBLE_COLUMN_LESS)
        .fetch_one(&pool)
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_create_leaves_no_table_for_the_next_commit_to_publish() {
    // A write's start used to insert the table's row, stamped with a guessed snapshot; a
    // refused commit left it, and the next commit, taking that snapshot, published a table
    // with no columns, which DuckDB will not open a lake over. The row is the commit's now.
    for inlining in [10, 0] {
        let temp = TempDir::new().unwrap();
        let writer = table_writer(&temp, inlining).await;
        writer
            .write_table("main", "existing", &[batch(1, "kept")])
            .await
            .unwrap();
        let mut late = writer
            .begin_write("main", "t", &schema(), WriteMode::Create)
            .unwrap();
        late.write_batch(&batch(9, "late")).unwrap();
        add_sqlite_view(&temp, "t").await;
        assert_view_exists(late.finish().await.unwrap_err());
        writer
            .create_table("main", "next", &[batch(2, "next")])
            .await
            .unwrap();
        assert_eq!(column_less_tables(&temp).await, 0, "inlining {inlining}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abandoned_write_leaves_no_table_for_the_next_commit_to_publish() {
    let temp = TempDir::new().unwrap();
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.db").display()
    );
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    abandon_a_write_then_commit(Arc::new(writer)).await;
    assert_eq!(column_less_tables(&temp).await, 0);
}

/// A SQLite catalog's writer, concrete and shared, and a table writer over it.
async fn sqlite_writers(temp: &TempDir) -> (Arc<SqliteMetadataWriter>, DuckLakeTableWriter) {
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let conn = format!(
        "sqlite:{}?mode=rwc",
        temp.path().join("catalog.db").display()
    );
    let metadata = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
    metadata.set_data_path(data_path.to_str().unwrap()).unwrap();
    let metadata = Arc::new(metadata);
    let shared: Arc<dyn MetadataWriter> = Arc::clone(&metadata) as Arc<dyn MetadataWriter>;
    let writer = DuckLakeTableWriter::new(shared, Arc::new(LocalFileSystem::new())).unwrap();
    (metadata, writer)
}

/// `(live rows, rows a reader sees at the head)` of `table`, in any schema.
async fn table_rows(temp: &TempDir, table: &str) -> (i64, i64) {
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    sqlx::query_as(
        "SELECT count(*), coalesce(sum(begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot)), 0)
         FROM ducklake_table WHERE table_name = ? AND end_snapshot IS NULL",
    )
    .bind(table)
    .fetch_one(&pool)
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn get_or_create_takes_up_a_pending_table_rather_than_making_a_second() {
    // A write in flight has made `t` (pending); creating `t` meanwhile takes that row up,
    // and the write then commits into it: one table named `t`, not two.
    let temp = TempDir::new().unwrap();
    let (metadata, writer) = sqlite_writers(&temp).await;
    let mut late = writer
        .begin_write("main", "t", &schema(), WriteMode::Append)
        .unwrap();
    late.write_batch(&batch(1, "late")).unwrap();
    let snapshot = metadata.create_snapshot().unwrap();
    let (main, _) = metadata
        .get_or_create_schema("main", None, snapshot)
        .unwrap();
    let (_, created) = metadata
        .get_or_create_table(main, "t", None, snapshot)
        .unwrap();
    assert!(
        created,
        "the pending table is created here, at this snapshot"
    );
    late.finish().await.unwrap();
    assert_eq!(
        table_rows(&temp, "t").await,
        (1, 1),
        "one live table t, visible"
    );
    assert_eq!(names(&temp).await, ["late"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abandoned_write_in_a_new_schema_publishes_nothing_and_its_retry_does() {
    let temp = TempDir::new().unwrap();
    let (_, writer) = sqlite_writers(&temp).await;
    {
        let mut abandoned = writer
            .begin_write("fresh", "t", &schema(), WriteMode::Append)
            .unwrap();
        abandoned.write_batch(&batch(1, "lost")).unwrap();
    }
    writer
        .create_table("main", "after", &[batch(2, "after")])
        .await
        .unwrap();
    assert_eq!(column_less_tables(&temp).await, 0);
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    let visible_fresh: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM ducklake_schema WHERE schema_name = 'fresh' AND end_snapshot IS NULL
           AND begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot)",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        visible_fresh, 0,
        "the abandoned write's new schema is no reader's"
    );

    // The retry takes the pending rows up: the schema and the table appear once.
    writer
        .create_table("fresh", "t", &[batch(3, "retried")])
        .await
        .unwrap();
    assert_eq!(table_rows(&temp, "t").await, (1, 1));
    let schemas: (i64, i64) = sqlx::query_as(
        "SELECT count(*), coalesce(sum(begin_snapshot <= (SELECT MAX(snapshot_id) FROM ducklake_snapshot)), 0)
         FROM ducklake_schema WHERE schema_name = 'fresh' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(schemas, (1, 1), "one schema fresh, visible");
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_a_table_that_is_only_pending_drops_nothing() {
    let temp = TempDir::new().unwrap();
    let (metadata, writer) = sqlite_writers(&temp).await;
    {
        let mut abandoned = writer
            .begin_write("main", "gone", &schema(), WriteMode::Append)
            .unwrap();
        abandoned.write_batch(&batch(1, "lost")).unwrap();
    }
    assert!(
        !metadata.drop_table("main", "gone").unwrap(),
        "a pending table is no table to drop"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pending_table_taken_up_gets_the_path_it_is_created_with() {
    let temp = TempDir::new().unwrap();
    let (metadata, writer) = sqlite_writers(&temp).await;
    {
        let mut abandoned = writer
            .begin_write("main", "p", &schema(), WriteMode::Append)
            .unwrap();
        abandoned.write_batch(&batch(1, "lost")).unwrap();
    }
    let snapshot = metadata.create_snapshot().unwrap();
    let (main, _) = metadata
        .get_or_create_schema("main", None, snapshot)
        .unwrap();
    let (_, created) = metadata
        .get_or_create_table(main, "p", Some("custom"), snapshot)
        .unwrap();
    assert!(created);
    let conn = format!("sqlite:{}", temp.path().join("catalog.db").display());
    let pool = sqlx::SqlitePool::connect(&conn).await.unwrap();
    let path: String = sqlx::query_scalar(
        "SELECT path FROM ducklake_table WHERE table_name = 'p' AND end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        path, "custom/",
        "the take-up records the path it is created with"
    );
}
