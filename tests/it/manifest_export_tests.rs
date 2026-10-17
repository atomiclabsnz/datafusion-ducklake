//! `DuckLakeTable::export_manifest`, checked against DuckDB.
//!
//! The oracle reads each snapshot with `AT (VERSION => n)`. The manifest side
//! reads the files the manifest lists, binds their columns by field id (or the
//! name mapping it reports), and drops the positions it lists as deleted. The
//! two must hold the same rows at every snapshot, whoever wrote the lake.

#![cfg(feature = "metadata-duckdb")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_ducklake::{
    ColumnBinding, DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckdbMetadataProvider,
    ManifestError, MetadataProvider, TableManifest,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

use crate::common::ensure_ducklake_installed;

pub(crate) type Rows = Vec<Vec<Option<String>>>;

/// A DuckDB connection with `lake` attached to the DuckDB catalog `catalog`.
pub(crate) fn attach(catalog: &Path, data: &Path, inlining_rows: usize) -> duckdb::Connection {
    ensure_ducklake_installed();
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "LOAD ducklake; ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', DATA_INLINING_ROW_LIMIT {inlining_rows});",
        catalog.display(),
        data.display()
    ))
    .unwrap();
    conn
}

pub(crate) fn snapshot_ids(conn: &duckdb::Connection) -> Vec<i64> {
    let mut statement = conn
        .prepare("SELECT snapshot_id FROM lake.snapshots() ORDER BY snapshot_id")
        .unwrap();
    statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// DuckDB's rows of `table` at `snapshot`, every value as text, or `None`
/// when the table does not exist there.
pub(crate) fn oracle_rows(conn: &duckdb::Connection, table: &str, snapshot: i64) -> Option<Rows> {
    let sql =
        format!("SELECT COLUMNS(*)::VARCHAR FROM lake.main.{table} AT (VERSION => {snapshot})");
    let mut statement = conn.prepare(&sql).ok()?;
    let mut result = statement.query([]).ok()?;
    let mut rows = Vec::new();
    while let Some(row) = result.next().unwrap() {
        let width = row.as_ref().column_count();
        rows.push(
            (0..width)
                .map(|index| row.get::<_, Option<String>>(index).unwrap())
                .collect(),
        );
    }
    rows.sort();
    Some(rows)
}

/// `table` at `snapshot`, opened through this crate, or `None` when it does
/// not exist there.
async fn table_at(
    provider: &Arc<dyn MetadataProvider>,
    table: &str,
    snapshot: i64,
) -> Option<Arc<dyn datafusion::catalog::TableProvider>> {
    let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(provider), snapshot).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    ctx.catalog("lake")?
        .schema("main")?
        .table(table)
        .await
        .unwrap()
}

pub(crate) async fn export(
    provider: &Arc<dyn MetadataProvider>,
    table: &str,
    snapshot: i64,
) -> Option<datafusion_ducklake::Result<TableManifest>> {
    let table = table_at(provider, table, snapshot).await?;
    let table = (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .unwrap();
    Some(table.export_manifest(&SessionContext::new().state()).await)
}

/// The rows a manifest describes: each listed file read directly, its columns
/// bound to the table's by field id or name mapping, its deleted positions
/// dropped.
pub(crate) fn manifest_rows(manifest: &TableManifest) -> Rows {
    let columns: Vec<(i64, &str)> = manifest
        .fields
        .iter()
        .filter(|field| field.parent_column.is_none())
        .map(|field| (field.column_id, field.column_name.as_str()))
        .collect();
    let mut rows = Vec::new();
    for file in &manifest.files {
        let local = file
            .path
            .strip_prefix("file://")
            .unwrap_or_else(|| panic!("{}", file.path));
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(local).unwrap()).unwrap();
        let schema = builder.schema().clone();
        let reader = builder.build().unwrap();
        let by_id: HashMap<i64, usize> = match &file.column_binding {
            ColumnBinding::FieldIds => schema
                .fields()
                .iter()
                .enumerate()
                .filter_map(|(index, field)| {
                    let id = field.metadata().get(PARQUET_FIELD_ID_META_KEY)?;
                    Some((id.parse().unwrap(), index))
                })
                .collect(),
            ColumnBinding::NameMapping(mapping) => mapping
                .entries
                .iter()
                .filter(|entry| entry.parent_column.is_none())
                .filter_map(|entry| {
                    let index = schema.index_of(&entry.source_name).ok()?;
                    Some((entry.target_field_id, index))
                })
                .collect(),
            other => panic!("{} is not bound: {other:?}", file.path),
        };
        let mut position = 0i64;
        for batch in reader {
            let batch = batch.unwrap();
            for row in 0..batch.num_rows() {
                if file.deleted_positions.binary_search(&position).is_err() {
                    rows.push(
                        columns
                            .iter()
                            .map(|(id, _)| {
                                let index = by_id.get(id)?;
                                let column = batch.column(*index);
                                column
                                    .is_valid(row)
                                    .then(|| array_value_to_string(column, row).unwrap())
                            })
                            .collect(),
                    );
                }
                position += 1;
            }
        }
        assert_eq!(position, file.record_count, "{}", file.path);
    }
    rows.sort();
    rows
}

/// At every snapshot of `lake.main.<table>`, the manifest holds DuckDB's rows.
/// Returns how many snapshots had the table.
async fn assert_every_snapshot_matches(catalog: &Path, data: &Path, table: &str) -> usize {
    let conn = attach(catalog, data, 0);
    let mut expected = Vec::new();
    for snapshot in snapshot_ids(&conn) {
        expected.push((snapshot, oracle_rows(&conn, table, snapshot)));
    }
    drop(conn);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let mut compared = 0;
    for (snapshot, oracle) in expected {
        let manifest = export(&provider, table, snapshot).await;
        match (oracle, manifest) {
            (None, None) => {},
            (Some(oracle), Some(manifest)) => {
                let manifest = manifest.unwrap();
                assert_eq!(manifest.snapshot_id, snapshot);
                assert_eq!(manifest.live_row_count(), oracle.len() as i64);
                assert_eq!(manifest_rows(&manifest), oracle, "snapshot {snapshot}");
                compared += 1;
            },
            (oracle, manifest) => panic!(
                "snapshot {snapshot}: DuckDB has the table: {}, this crate: {}",
                oracle.is_some(),
                manifest.is_some()
            ),
        }
    }
    compared
}

/// Deletes (twice to one file, so the second delete file records each
/// deletion's snapshot), schema evolution, a rename, an update and a merge of
/// files from several snapshots.
#[tokio::test(flavor = "multi_thread")]
async fn duckdb_written_lake_matches_at_every_snapshot() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    let conn = attach(&catalog, &data, 0);
    conn.execute_batch(
        "CREATE TABLE lake.t (id INT, name VARCHAR);
         INSERT INTO lake.t SELECT i, 'n' || i FROM range(0, 10) r(i);
         INSERT INTO lake.t SELECT i, 'n' || i FROM range(10, 20) r(i);
         DELETE FROM lake.t WHERE id % 3 = 0;
         DELETE FROM lake.t WHERE id = 4;
         ALTER TABLE lake.t ADD COLUMN extra INT;
         INSERT INTO lake.t SELECT i, 'n' || i, i * 10 FROM range(20, 25) r(i);
         ALTER TABLE lake.t RENAME COLUMN name TO label;
         UPDATE lake.t SET extra = -1 WHERE id = 21;
         INSERT INTO lake.t SELECT i, 'n' || i, i FROM range(25, 28) r(i);
         CALL ducklake_merge_adjacent_files('lake');
         DELETE FROM lake.t WHERE id IN (22, 26);",
    )
    .unwrap();
    drop(conn);

    let compared = assert_every_snapshot_matches(&catalog, &data, "t").await;
    assert!(compared >= 10, "{compared}");
}

/// A merge of files from several snapshots writes one file, beginning at the
/// first, whose rows carry their snapshot. Read at an earlier snapshot, the
/// rows inserted after it are listed as deleted.
#[tokio::test(flavor = "multi_thread")]
async fn merged_file_rows_after_the_snapshot_are_deleted_positions() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    let conn = attach(&catalog, &data, 0);
    conn.execute_batch(
        "CREATE TABLE lake.t (id INT);
         INSERT INTO lake.t VALUES (1), (2);
         INSERT INTO lake.t VALUES (3);
         INSERT INTO lake.t VALUES (4), (5);
         CALL ducklake_merge_adjacent_files('lake');",
    )
    .unwrap();
    drop(conn);

    assert!(assert_every_snapshot_matches(&catalog, &data, "t").await >= 3);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    // Snapshot 0 creates the schema, 1 the table, 2 the first insert.
    let first_insert = 2;
    let manifest = export(&provider, "t", first_insert).await.unwrap().unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.files[0].record_count, 5);
    assert_eq!(manifest.files[0].deleted_positions, vec![2, 3, 4]);
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_lake_reports_spec_and_values() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    let conn = attach(&catalog, &data, 0);
    conn.execute_batch(
        "CREATE TABLE lake.p (id INT, part INT);
         ALTER TABLE lake.p SET PARTITIONED BY (part);
         INSERT INTO lake.p SELECT i, i % 3 FROM range(0, 12) r(i);
         DELETE FROM lake.p WHERE id IN (1, 5);",
    )
    .unwrap();
    drop(conn);

    assert!(assert_every_snapshot_matches(&catalog, &data, "p").await >= 2);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let manifest = export(&provider, "p", head).await.unwrap().unwrap();
    let spec = manifest.partition_spec.clone().expect("partitioned");
    assert_eq!(manifest.files.len(), 3);
    let mut values: Vec<_> = manifest
        .files
        .iter()
        .map(|file| {
            assert_eq!(file.partition_id, Some(spec.partition_id));
            file.partition_values.clone()
        })
        .collect();
    values.sort();
    assert_eq!(
        values,
        vec![
            vec![(0, Some("0".to_string()))],
            vec![(0, Some("1".to_string()))],
            vec![(0, Some("2".to_string()))],
        ]
    );
    assert!(
        manifest
            .files
            .iter()
            .all(|file| !file.column_statistics.is_empty())
    );
}

/// Inlined rows are in no file, so the export refuses them until they are
/// flushed; after the flush every snapshot matches.
#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_refused_until_flushed() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    let conn = attach(&catalog, &data, 10);
    conn.execute_batch(
        "CREATE TABLE lake.t (id INT, name VARCHAR);
         INSERT INTO lake.t VALUES (1, 'a'), (2, 'b');",
    )
    .unwrap();
    drop(conn);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let refused = export(&provider, "t", head).await.unwrap().unwrap_err();
    assert!(
        matches!(
            refused,
            DuckLakeError::Manifest(ManifestError::InlinedRows {
                rows: 2,
                ..
            })
        ),
        "{refused}"
    );
    drop(provider);

    let conn = attach(&catalog, &data, 10);
    conn.execute_batch("CALL ducklake_flush_inlined_data('lake');")
        .unwrap();
    drop(conn);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let manifest = export(&provider, "t", head).await.unwrap().unwrap();
    assert_eq!(manifest.live_row_count(), 2);
    drop(provider);
    assert!(assert_every_snapshot_matches(&catalog, &data, "t").await >= 1);
}

/// A file added with `ducklake_add_data_files` has no field ids; the manifest
/// carries the catalog's name mapping for it. Without that mapping nothing
/// binds its columns, and the manifest says so rather than guessing.
#[tokio::test(flavor = "multi_thread")]
async fn files_without_field_ids_carry_their_mapping_or_are_reported() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let external = temp.path().join("external.parquet");
    let conn = attach(&catalog, &data, 0);
    conn.execute_batch(&format!(
        "CREATE TABLE lake.t (id INT, name VARCHAR);
         INSERT INTO lake.t VALUES (1, 'a');
         COPY (SELECT 2 AS id, 'b' AS name UNION ALL SELECT 3, 'c') TO '{}' (FORMAT PARQUET);
         CALL ducklake_add_data_files('lake', 't', '{}');
         DELETE FROM lake.t WHERE id = 3;",
        external.display(),
        external.display()
    ))
    .unwrap();
    drop(conn);

    assert!(assert_every_snapshot_matches(&catalog, &data, "t").await >= 3);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let manifest = export(&provider, "t", head).await.unwrap().unwrap();
    let added = manifest
        .files
        .iter()
        .find(|file| file.path.ends_with("external.parquet"))
        .unwrap();
    assert!(matches!(
        added.column_binding,
        ColumnBinding::NameMapping(_)
    ));
    assert_eq!(added.deleted_positions, vec![1]);
    assert_eq!(manifest.unresolved_files().count(), 0);
    drop(provider);

    let conn = duckdb::Connection::open(&catalog).unwrap();
    conn.execute_batch("UPDATE ducklake_data_file SET mapping_id = NULL")
        .unwrap();
    drop(conn);
    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let manifest = export(&provider, "t", head).await.unwrap().unwrap();
    let unresolved: Vec<_> = manifest.unresolved_files().collect();
    assert_eq!(unresolved.len(), 1);
    assert_eq!(
        unresolved[0].column_binding,
        ColumnBinding::Unresolved {
            columns: vec!["id".to_string(), "name".to_string()]
        }
    );
}

/// The keys of encrypted files are in the catalog, so another format cannot
/// read them in place.
#[tokio::test(flavor = "multi_thread")]
async fn encrypted_files_are_refused() {
    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    ensure_ducklake_installed();
    let conn = duckdb::Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "LOAD ducklake; ATTACH 'ducklake:{}' AS lake (DATA_PATH '{}', ENCRYPTED, DATA_INLINING_ROW_LIMIT 0);
         CREATE TABLE lake.t (id INT);
         INSERT INTO lake.t VALUES (1), (2);",
        catalog.display(),
        data.display()
    ))
    .unwrap();
    drop(conn);

    let provider: Arc<dyn MetadataProvider> =
        Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
    let head = provider.get_current_snapshot().unwrap();
    let refused = export(&provider, "t", head).await.unwrap().unwrap_err();
    assert!(
        matches!(
            refused,
            DuckLakeError::Manifest(ManifestError::EncryptedFile { .. })
        ),
        "{refused}"
    );
}

/// A lake this crate wrote: two files, deletes from both, the second delete of
/// one file superseding its first delete file, and an appended column.
#[cfg(feature = "write-duckdb")]
#[tokio::test(flavor = "multi_thread")]
async fn crate_written_lake_matches_at_every_snapshot() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_ducklake::{DuckLakeTableWriter, DuckdbMetadataWriter, MetadataWriter};
    use object_store::local::LocalFileSystem;

    let temp = TempDir::new().unwrap();
    let catalog = temp.path().join("catalog.ducklake");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    {
        let writer = Arc::new(
            DuckdbMetadataWriter::new_with_init(catalog.to_str().unwrap())
                .unwrap()
                .with_data_inlining_row_limit(0),
        );
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        let table_writer =
            DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
        let two = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = |ids: std::ops::Range<i32>| {
            RecordBatch::try_new(
                two.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.clone().collect::<Vec<_>>())),
                    Arc::new(StringArray::from(
                        ids.map(|id| format!("n{id}")).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()
        };
        table_writer
            .write_table("main", "t", &[batch(0..10)])
            .await
            .unwrap();
        table_writer
            .append_table("main", "t", &[batch(10..20)])
            .await
            .unwrap();

        for statement in
            ["DELETE FROM lake.main.t WHERE id % 4 = 0", "DELETE FROM lake.main.t WHERE id = 5"]
        {
            let provider: Arc<dyn MetadataProvider> =
                Arc::new(DuckdbMetadataProvider::new(catalog.to_string_lossy()).unwrap());
            let lake = DuckLakeCatalog::with_writer(provider, writer.clone()).unwrap();
            let ctx = SessionContext::new();
            ctx.register_catalog("lake", Arc::new(lake));
            ctx.sql(statement).await.unwrap().collect().await.unwrap();
        }

        let three = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("extra", DataType::Int32, true),
        ]));
        let extended = RecordBatch::try_new(
            three,
            vec![
                Arc::new(Int32Array::from(vec![20, 21])),
                Arc::new(StringArray::from(vec!["n20", "n21"])),
                Arc::new(Int32Array::from(vec![Some(7), None])),
            ],
        )
        .unwrap();
        table_writer
            .append_table("main", "t", &[extended])
            .await
            .unwrap();
    }

    let compared = assert_every_snapshot_matches(&catalog, &data, "t").await;
    assert!(compared >= 5, "{compared}");
}
