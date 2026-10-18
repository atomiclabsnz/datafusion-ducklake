#![cfg(all(feature = "write-sqlite", feature = "write"))]
//! `WriteMode::Create`, the write of `CREATE TABLE … AS`: a table that exists is refused, and
//! so is one another writer creates after this write began, inside the commit's own
//! transaction. Each case runs with the rows inlined in the catalog and with them in Parquet,
//! the two ways a write commits.

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

/// The three Create cases on a backend's own writer: a table that exists, a table another
/// writer creates while this one is writing, and a new table. The commit's check is each
/// backend's own code, so each runs it.
async fn create_refusals(metadata: Arc<dyn MetadataWriter>) {
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
    create_refusals(Arc::new(writer)).await;
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
    create_refusals(Arc::new(writer)).await;
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
    create_refusals(Arc::new(single)).await;

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
    let multi = datafusion_ducklake::PostgresMetadataWriter::with_pool(pool, catalog_id)
        .await
        .unwrap();
    let temp = TempDir::new().unwrap();
    multi.set_data_path(&data_path(&temp)).unwrap();
    create_refusals(Arc::new(multi)).await;
}
