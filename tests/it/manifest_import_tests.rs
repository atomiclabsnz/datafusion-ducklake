//! `DuckLakeTableWriter::import_manifest`: existing Parquet files plus deleted
//! positions registered as one snapshot of a SQLite-catalog table.
//!
//! DuckDB is the oracle both ways: the imported table must hold the source's
//! rows when this crate reads it and when DuckDB attaches the catalog.

#![cfg(all(feature = "write-sqlite", feature = "metadata-duckdb"))]

use std::path::Path;
use std::sync::Arc;

use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::maintenance::{
    CleanupCriteria, ExpireCriteria, cleanup_old_files_sqlite, delete_orphaned_files_sqlite,
};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, DuckdbMetadataProvider, ImportError,
    ImportOptions, MetadataProvider, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    TableManifest,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use crate::common::{ensure_ducklake_installed, ensure_extension_installed};
use crate::manifest_export_tests::{Rows, attach, export, oracle_rows};

/// A DuckDB-written lake, built by `script`, under `dir/source`.
struct Source {
    catalog: std::path::PathBuf,
    data: std::path::PathBuf,
}

impl Source {
    fn new(dir: &Path, script: &str) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        let catalog = dir.join("source.ducklake");
        let data = dir.join("source");
        let conn = attach(&catalog, &data, 0);
        conn.execute_batch(script).unwrap();
        Self {
            catalog,
            data,
        }
    }

    fn provider(&self) -> Arc<dyn MetadataProvider> {
        Arc::new(DuckdbMetadataProvider::new(self.catalog.to_string_lossy()).unwrap())
    }

    async fn manifest(&self, table: &str, snapshot: i64) -> TableManifest {
        export(&self.provider(), table, snapshot)
            .await
            .unwrap()
            .unwrap()
    }

    async fn head_manifest(&self, table: &str) -> TableManifest {
        let head = self.provider().get_current_snapshot().unwrap();
        self.manifest(table, head).await
    }

    fn rows(&self, table: &str, snapshot: Option<i64>) -> Rows {
        let conn = attach(&self.catalog, &self.data, 0);
        let snapshot = snapshot.unwrap_or_else(|| {
            conn.query_row("SELECT max(snapshot_id) FROM lake.snapshots()", [], |row| {
                row.get(0)
            })
            .unwrap()
        });
        oracle_rows(&conn, table, snapshot).unwrap()
    }
}

/// A SQLite catalog this crate writes, under `dir/target`.
struct Target {
    catalog: std::path::PathBuf,
    data: std::path::PathBuf,
}

impl Target {
    async fn new(dir: &Path) -> Self {
        let catalog = dir.join("target.sqlite");
        let data = dir.join("target");
        std::fs::create_dir_all(&data).unwrap();
        let target = Self {
            catalog,
            data,
        };
        let writer = SqliteMetadataWriter::new_with_init(&target.url())
            .await
            .unwrap();
        writer.set_data_path(target.data.to_str().unwrap()).unwrap();
        target
    }

    fn url(&self) -> String {
        format!("sqlite:{}?mode=rwc", self.catalog.display())
    }

    async fn writer(&self) -> Arc<SqliteMetadataWriter> {
        Arc::new(
            SqliteMetadataWriter::new(&self.url())
                .await
                .unwrap()
                .with_data_inlining_row_limit(0),
        )
    }

    async fn provider(&self) -> Arc<dyn MetadataProvider> {
        Arc::new(SqliteMetadataProvider::new(&self.url()).await.unwrap())
    }

    async fn table_writer(&self) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(self.writer().await, Arc::new(LocalFileSystem::new())).unwrap()
    }

    async fn import(
        &self,
        table: &str,
        manifest: &TableManifest,
        options: &ImportOptions,
    ) -> datafusion_ducklake::Result<datafusion_ducklake::ImportResult> {
        self.table_writer()
            .await
            .import_manifest(self.provider().await, "main", table, manifest, options)
            .await
    }

    /// The table's rows as this crate reads them, every value as text.
    async fn rows(&self, table: &str) -> Rows {
        let ctx = SessionContext::new();
        let provider = SqliteMetadataProvider::new(&self.url()).await.unwrap();
        let catalog = DuckLakeCatalog::new(provider).unwrap();
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx
            .sql(&format!("SELECT * FROM lake.main.{table}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                rows.push(
                    batch
                        .columns()
                        .iter()
                        .map(|column| {
                            column
                                .is_valid(row)
                                .then(|| array_value_to_string(column, row).unwrap())
                        })
                        .collect(),
                );
            }
        }
        rows.sort();
        rows
    }

    /// The table's rows as DuckDB reads them, through a copy of the catalog
    /// given the columns and tables DuckDB requires that this crate's SQLite
    /// catalogs omit.
    async fn duckdb_rows(&self, table: &str) -> Rows {
        let copy = self.catalog.with_extension("duckdb-view.sqlite");
        std::fs::copy(&self.catalog, &copy).unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", copy.display()))
            .await
            .unwrap();
        for statement in [
            "ALTER TABLE ducklake_snapshot ADD COLUMN next_catalog_id BIGINT NOT NULL DEFAULT 1000",
            "ALTER TABLE ducklake_snapshot ADD COLUMN next_file_id BIGINT NOT NULL DEFAULT 1000",
            "ALTER TABLE ducklake_schema ADD COLUMN schema_uuid VARCHAR",
            "UPDATE ducklake_schema SET schema_uuid = '00000000-0000-0000-0000-000000000001'",
            "ALTER TABLE ducklake_table ADD COLUMN table_uuid VARCHAR",
            "UPDATE ducklake_table SET table_uuid = '00000000-0000-0000-0000-000000000002'",
            "ALTER TABLE ducklake_data_file ADD COLUMN file_order BIGINT",
            "ALTER TABLE ducklake_data_file ADD COLUMN file_format VARCHAR",
            "ALTER TABLE ducklake_delete_file ADD COLUMN format VARCHAR",
            "CREATE TABLE ducklake_file_variant_stats (data_file_id BIGINT, table_id BIGINT, column_id BIGINT, variant_path VARCHAR, shredded_type VARCHAR, column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT, min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN, extra_stats VARCHAR)",
            "CREATE TABLE ducklake_macro (schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT)",
            "CREATE TABLE ducklake_macro_impl (macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR)",
            "CREATE TABLE ducklake_macro_parameters (macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR)",
            "CREATE TABLE IF NOT EXISTS ducklake_view (view_id BIGINT, view_uuid VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT, schema_id BIGINT, view_name VARCHAR, dialect VARCHAR, sql VARCHAR, column_aliases VARCHAR)",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        pool.close().await;

        ensure_ducklake_installed();
        ensure_extension_installed("sqlite");
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "LOAD ducklake; LOAD sqlite; ATTACH 'ducklake:sqlite:{}' AS lake (READ_ONLY);",
            copy.display()
        ))
        .unwrap();
        let snapshot: i64 = conn
            .query_row("SELECT max(snapshot_id) FROM lake.snapshots()", [], |row| {
                row.get(0)
            })
            .unwrap();
        oracle_rows(&conn, table, snapshot).unwrap()
    }
}

const EVOLVED: &str = "CREATE TABLE lake.t (id INT, name VARCHAR);
     INSERT INTO lake.t SELECT i, 'n' || i FROM range(0, 10) r(i);
     INSERT INTO lake.t SELECT i, 'n' || i FROM range(10, 20) r(i);
     DELETE FROM lake.t WHERE id % 3 = 0;
     DELETE FROM lake.t WHERE id = 4;
     ALTER TABLE lake.t ADD COLUMN extra INT;
     INSERT INTO lake.t SELECT i, 'n' || i, i * 10 FROM range(20, 25) r(i);
     ALTER TABLE lake.t RENAME COLUMN name TO label;
     DELETE FROM lake.t WHERE id = 22;";

/// Files DuckDB wrote, with deletes from two statements, a later column and a
/// rename, read back by both engines after the import.
#[tokio::test(flavor = "multi_thread")]
async fn imported_duckdb_files_read_back_in_both_engines() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(dir.path(), EVOLVED);
    let manifest = source.head_manifest("t").await;
    let expected = source.rows("t", None);
    assert!(
        manifest
            .files
            .iter()
            .any(|f| !f.deleted_positions.is_empty())
    );

    let target = Target::new(dir.path()).await;
    let result = target
        .import("t", &manifest, &ImportOptions::default())
        .await
        .unwrap();
    assert_eq!(result.files, manifest.files.len());
    assert_eq!(
        result.deleted_rows,
        manifest
            .files
            .iter()
            .map(|f| f.deleted_positions.len() as i64)
            .sum::<i64>()
    );

    assert_eq!(target.rows("t").await, expected);
    assert_eq!(target.duckdb_rows("t").await, expected);

    // One snapshot holds every file and every delete: the one before it has
    // no table, so no snapshot ever shows the deleted rows.
    let provider = target.provider().await;
    assert_eq!(provider.get_current_snapshot().unwrap(), result.snapshot_id);
    let before = DuckLakeCatalog::with_snapshot(provider, result.snapshot_id - 1).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(before));
    assert!(ctx.sql("SELECT * FROM lake.main.t").await.is_err());

    // The imported table exports the same rows again.
    let reexported = export(&target.provider().await, "t", result.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        crate::manifest_export_tests::manifest_rows(&reexported),
        expected
    );
}

/// A merged file read at a snapshot before its last rows: those rows arrive
/// as deleted positions, and the import shows exactly that snapshot's rows.
#[tokio::test(flavor = "multi_thread")]
async fn importing_an_older_snapshot_of_a_merged_file() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(
        dir.path(),
        "CREATE TABLE lake.t (id INT);
         INSERT INTO lake.t VALUES (1), (2);
         INSERT INTO lake.t VALUES (3);
         INSERT INTO lake.t VALUES (4), (5);
         CALL ducklake_merge_adjacent_files('lake');",
    );
    let manifest = source.manifest("t", 3).await;
    assert_eq!(manifest.files[0].deleted_positions, vec![3, 4]);

    let target = Target::new(dir.path()).await;
    target
        .import("t", &manifest, &ImportOptions::default())
        .await
        .unwrap();
    let expected = source.rows("t", Some(3));
    assert_eq!(target.rows("t").await, expected);
    assert_eq!(target.duckdb_rows("t").await, expected);
}

/// A `ducklake_add_data_files` file has no field ids; its name mapping comes
/// along and binds it in the target.
#[tokio::test(flavor = "multi_thread")]
async fn name_mapped_files_keep_their_mapping() {
    let dir = TempDir::new().unwrap();
    let external = dir.path().join("external.parquet");
    let source = Source::new(
        dir.path(),
        &format!(
            "CREATE TABLE lake.t (id INT, name VARCHAR);
             COPY (SELECT 2 AS id, 'b' AS name UNION ALL SELECT 3, 'c') TO '{0}' (FORMAT PARQUET);
             CALL ducklake_add_data_files('lake', 't', '{0}');
             DELETE FROM lake.t WHERE id = 3;",
            external.display()
        ),
    );
    let manifest = source.head_manifest("t").await;
    let target = Target::new(dir.path()).await;
    target
        .import("t", &manifest, &ImportOptions::default())
        .await
        .unwrap();
    let expected = source.rows("t", None);
    assert_eq!(expected, vec![vec![Some("2".into()), Some("b".into())]]);
    assert_eq!(target.rows("t").await, expected);
    assert_eq!(target.duckdb_rows("t").await, expected);
}

/// A lake this crate wrote, exported and imported into another catalog.
#[tokio::test(flavor = "multi_thread")]
async fn crate_written_files_round_trip() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let dir = TempDir::new().unwrap();
    let first = Target::new(&dir.path().join("first")).await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
        ],
    )
    .unwrap();
    first
        .table_writer()
        .await
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
    let lake = DuckLakeCatalog::with_writer(first.provider().await, first.writer().await).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(lake));
    ctx.sql("DELETE FROM lake.main.t WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let provider = first.provider().await;
    let head = provider.get_current_snapshot().unwrap();
    let manifest = export(&provider, "t", head).await.unwrap().unwrap();
    assert_eq!(manifest.files[0].deleted_positions, vec![1]);

    let second = Target::new(&dir.path().join("second")).await;
    second
        .import("copy", &manifest, &ImportOptions::default())
        .await
        .unwrap();
    let expected = first.rows("t").await;
    assert_eq!(expected.len(), 3);
    assert_eq!(second.rows("copy").await, expected);
    assert_eq!(second.duckdb_rows("copy").await, expected);
}

fn import_error(error: DuckLakeError) -> ImportError {
    match error {
        DuckLakeError::Import(error) => error,
        other => panic!("not an import error: {other}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_field_ids_are_refused_per_file() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(dir.path(), EVOLVED);
    let mut manifest = source.head_manifest("t").await;
    // The table's `label` column, under an id no file uses for it.
    let label = manifest
        .fields
        .iter_mut()
        .find(|field| field.column_name == "label")
        .unwrap();
    let old_id = label.column_id;
    label.column_id += 100;
    let target = Target::new(dir.path()).await;
    let head = target.provider().await.get_current_snapshot().unwrap();
    let error = import_error(
        target
            .import("t", &manifest, &ImportOptions::default())
            .await
            .unwrap_err(),
    );
    assert!(
        matches!(
            &error,
            ImportError::FieldIdMismatch { column, field_id, .. }
                if column == "name" && *field_id == old_id
        ),
        "{error}"
    );
    // Nothing was committed.
    assert_eq!(
        target.provider().await.get_current_snapshot().unwrap(),
        head
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_position_past_the_rows_is_refused() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(dir.path(), EVOLVED);
    let mut manifest = source.head_manifest("t").await;
    let file = &mut manifest.files[0];
    let rows = file.record_count;
    file.deleted_positions.push(rows);
    let target = Target::new(dir.path()).await;
    let error = import_error(
        target
            .import("t", &manifest, &ImportOptions::default())
            .await
            .unwrap_err(),
    );
    assert!(
        matches!(
            error,
            ImportError::PositionOutOfRange { position, record_count, .. }
                if position == rows && record_count == rows
        ),
        "{error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_duplicate_file_is_refused() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(dir.path(), EVOLVED);
    let manifest = source.head_manifest("t").await;
    let target = Target::new(dir.path()).await;

    let mut twice = manifest.clone();
    twice.files.push(twice.files[0].clone());
    let error = import_error(
        target
            .import("t", &twice, &ImportOptions::default())
            .await
            .unwrap_err(),
    );
    assert!(
        matches!(error, ImportError::DuplicateFile { .. }),
        "{error}"
    );

    target
        .import("t", &manifest, &ImportOptions::default())
        .await
        .unwrap();
    let again = target
        .import("t", &manifest, &ImportOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(import_error(again), ImportError::DuplicateFile { .. }),
        "a file already in the table"
    );
    assert_eq!(target.rows("t").await, source.rows("t", None));
}

/// A writer that deletes from the table after the snapshot the import checked
/// it at makes the import's commit conflict, as an insert's would.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_delete_conflicts() {
    let dir = TempDir::new().unwrap();
    let first = Source::new(
        &dir.path().join("a"),
        "CREATE TABLE lake.t (id INT); INSERT INTO lake.t VALUES (1), (2);",
    );
    let second = Source::new(
        &dir.path().join("b"),
        "CREATE TABLE lake.t (id INT); INSERT INTO lake.t VALUES (3), (4);",
    );
    let first = first.head_manifest("t").await;
    let mut second = second.head_manifest("t").await;
    second.fields = first.fields.clone();

    let target = Target::new(dir.path()).await;
    target
        .import("t", &first, &ImportOptions::default())
        .await
        .unwrap();
    let base = target.provider().await.get_current_snapshot().unwrap();

    let lake =
        DuckLakeCatalog::with_writer(target.provider().await, target.writer().await).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(lake));
    ctx.sql("DELETE FROM lake.main.t WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let error = target
        .import(
            "t",
            &second,
            &ImportOptions::default().with_base_snapshot(base),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");
    assert_eq!(target.rows("t").await, vec![vec![Some("2".to_string())]]);

    // Checked at the current head, the same import commits.
    target
        .import("t", &second, &ImportOptions::default())
        .await
        .unwrap();
    assert_eq!(target.rows("t").await.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_existing_table_with_other_columns_is_refused() {
    let dir = TempDir::new().unwrap();
    let source = Source::new(dir.path(), EVOLVED);
    let manifest = source.head_manifest("t").await;
    let other = Source::new(
        &dir.path().join("other"),
        "CREATE TABLE lake.t (id INT); INSERT INTO lake.t VALUES (1);",
    );
    let target = Target::new(dir.path()).await;
    target
        .import(
            "t",
            &other.head_manifest("t").await,
            &ImportOptions::default(),
        )
        .await
        .unwrap();
    let error = import_error(
        target
            .import("t", &manifest, &ImportOptions::default())
            .await
            .unwrap_err(),
    );
    assert!(
        matches!(error, ImportError::SchemaMismatch { .. }),
        "{error}"
    );
}

/// Imported objects stay where the source table reads them: retiring them,
/// expiring every snapshot that read them and cleaning up leaves them in
/// place, unless the import took ownership.
#[tokio::test(flavor = "multi_thread")]
async fn cleanup_deletes_imported_files_only_when_owned() {
    for take_ownership in [false, true] {
        let dir = TempDir::new().unwrap();
        let source = Source::new(dir.path(), EVOLVED);
        let manifest = source.head_manifest("t").await;
        let target = Target::new(dir.path()).await;
        let expected = source.rows("t", None);
        target
            .import(
                "t",
                &manifest,
                &ImportOptions::default().with_take_ownership(take_ownership),
            )
            .await
            .unwrap();

        // Replace the table's rows, retiring every imported file.
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, true),
            arrow::datatypes::Field::new("label", arrow::datatypes::DataType::Utf8, true),
            arrow::datatypes::Field::new("extra", arrow::datatypes::DataType::Int32, true),
        ]));
        let batch = arrow::record_batch::RecordBatch::new_empty(schema);
        target
            .table_writer()
            .await
            .write_table("main", "t", &[batch])
            .await
            .unwrap();
        let writer = target.writer().await;
        writer
            .expire_snapshots(ExpireCriteria::OlderThan(
                chrono::Utc::now() + chrono::Duration::seconds(1),
            ))
            .unwrap();
        let store = Arc::new(LocalFileSystem::new());
        let cleaned = cleanup_old_files_sqlite(&writer, store.clone(), CleanupCriteria::All, false)
            .await
            .unwrap();
        delete_orphaned_files_sqlite(&writer, store, CleanupCriteria::All, false)
            .await
            .unwrap();
        assert!(!cleaned.is_empty() || !take_ownership, "{cleaned:?}");

        for file in &manifest.files {
            let local = file.path.strip_prefix("file://").unwrap();
            assert_eq!(
                Path::new(local).exists(),
                !take_ownership,
                "{local}, take_ownership = {take_ownership}"
            );
        }
        if !take_ownership {
            assert_eq!(source.rows("t", None), expected);
        }
    }
}

/// A new table is created unpartitioned; once it has the manifest's partition
/// spec, imported files keep their partition values under it.
#[tokio::test(flavor = "multi_thread")]
async fn partition_values_are_kept_under_a_matching_spec() {
    let dir = TempDir::new().unwrap();
    let script = |from: i32| {
        format!(
            "CREATE TABLE lake.p (id INT, part INT);
             ALTER TABLE lake.p SET PARTITIONED BY (part);
             INSERT INTO lake.p SELECT i, i % 2 FROM range({from}, {}) r(i);
             DELETE FROM lake.p WHERE id = {from};",
            from + 6
        )
    };
    let first = Source::new(&dir.path().join("a"), &script(0));
    let second = Source::new(&dir.path().join("b"), &script(100));
    let first_manifest = first.head_manifest("p").await;
    let second_manifest = second.head_manifest("p").await;
    assert!(second_manifest.partition_spec.is_some());

    let target = Target::new(dir.path()).await;
    let created = target
        .import("p", &first_manifest, &ImportOptions::default())
        .await
        .unwrap();
    let provider = target.provider().await;
    let manifest = export(&provider, "p", created.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    assert!(manifest.partition_spec.is_none());

    target
        .writer()
        .await
        .set_partition_spec(
            created.table_id,
            &[(
                "part".to_string(),
                datafusion_ducklake::PartitionTransform::Identity,
            )],
        )
        .unwrap();
    let imported = target
        .import("p", &second_manifest, &ImportOptions::default())
        .await
        .unwrap();
    let manifest = export(&target.provider().await, "p", imported.snapshot_id)
        .await
        .unwrap()
        .unwrap();
    let spec = manifest.partition_spec.clone().unwrap();
    let partitioned: Vec<_> = manifest
        .files
        .iter()
        .filter(|file| file.partition_id == Some(spec.partition_id))
        .map(|file| file.partition_values.clone())
        .collect();
    assert_eq!(partitioned.len(), 2, "{manifest:?}");

    let mut expected = first.rows("p", None);
    expected.extend(second.rows("p", None));
    expected.sort();
    assert_eq!(target.rows("p").await, expected);
    assert_eq!(target.duckdb_rows("p").await, expected);
    // Pruned by partition: only part = 1 rows.
    let ctx = SessionContext::new();
    let provider = SqliteMetadataProvider::new(&target.url()).await.unwrap();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    let ones = ctx
        .sql("SELECT count(*) FROM lake.main.p WHERE part = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ones = array_value_to_string(ones[0].column(0), 0).unwrap();
    let expected_ones = expected
        .iter()
        .filter(|row| row[1].as_deref() == Some("1"))
        .count();
    assert_eq!(ones, expected_ones.to_string());
}
