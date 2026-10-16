//! Export one snapshot of a table as a format-neutral file manifest.
//!
//! A [`TableManifest`] lists what a reader needs to commit the same Parquet
//! files into another table format (Iceberg, Delta) without rewriting them: the
//! table's columns with their `column_id`s, which are the Parquet `field_id`s of
//! the files, and for each live data file its location, sizes, catalog
//! statistics, partition values and the row positions deleted in it at the
//! snapshot.
//!
//! The deleted positions are resolved, not the delete files: a position is
//! listed when the snapshot does not show its row, whether a delete file, an
//! inlined deletion or a merged file's per-row snapshot id hides it. No target
//! format can represent per-row snapshot ids, so this is the only form they all
//! accept.
//!
//! The export reads and never writes. It refuses a table with inlined rows at
//! the snapshot, and a table with encrypted data files; see [`ManifestError`].

use std::collections::{HashMap, HashSet};

use arrow::datatypes::Field;
use datafusion::catalog::Session;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::Result;
use crate::metadata_provider::{
    DuckLakeFileColumnStatistics, DuckLakeNameMapping, DuckLakeTableField, DuckLakeTableFile,
};
use crate::partition::PartitionSpec;
use crate::table::{DuckLakeTable, read_parquet_footer_facts};

/// One snapshot of a table, as files another table format can adopt in place.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TableManifest {
    /// Catalog `table_id`.
    pub table_id: i64,
    /// The snapshot this manifest describes.
    pub snapshot_id: i64,
    /// The table's columns at the snapshot, nested ones included, in catalog
    /// order. A column's `column_id` is the `field_id` the data files carry for
    /// it, unless the file's [`ColumnBinding`] says otherwise.
    pub fields: Vec<DuckLakeTableField>,
    /// The partition spec live at the snapshot, or `None` when the table is
    /// unpartitioned there.
    pub partition_spec: Option<PartitionSpec>,
    /// Every data file live at the snapshot, by ascending `data_file_id`.
    pub files: Vec<ManifestDataFile>,
}

impl TableManifest {
    /// The files whose columns this manifest could not bind to the table's
    /// column ids ([`ColumnBinding::Unresolved`]).
    pub fn unresolved_files(&self) -> impl Iterator<Item = &ManifestDataFile> {
        self.files
            .iter()
            .filter(|file| matches!(file.column_binding, ColumnBinding::Unresolved { .. }))
    }

    /// The rows the snapshot shows: every file's record count less its deleted
    /// positions.
    pub fn live_row_count(&self) -> i64 {
        self.files
            .iter()
            .map(|file| file.record_count - file.deleted_positions.len() as i64)
            .sum()
    }
}

/// One live data file of a [`TableManifest`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ManifestDataFile {
    /// Catalog `data_file_id`.
    pub data_file_id: i64,
    /// The file's absolute location, resolved against the catalog's
    /// `data_path`, schema path and table path: `file:///…`, `s3://bucket/…`.
    pub path: String,
    /// Size of the file in bytes.
    pub file_size_bytes: i64,
    /// Rows the file holds, before any deletion.
    pub record_count: i64,
    /// Size of the Parquet footer in bytes, when the catalog records it.
    pub footer_size: Option<i64>,
    /// The catalog's per-column statistics for the file, keyed by `column_id`.
    /// Bounds are DuckDB-canonical text, as `ducklake_file_column_stats` stores
    /// them. They describe the whole file, deleted rows included.
    pub column_statistics: Vec<DuckLakeFileColumnStatistics>,
    /// The partition spec generation the file was written under. When it is
    /// not [`TableManifest::partition_spec`]'s `partition_id`, the file
    /// predates a re-partition and its values follow that older spec.
    pub partition_id: Option<i64>,
    /// The file's partition values as `(partition_key_index, value)`,
    /// DuckDB-canonical text, `None` for SQL NULL.
    pub partition_values: Vec<(i32, Option<String>)>,
    /// How the file's physical columns map onto the table's column ids.
    pub column_binding: ColumnBinding,
    /// Positions (0-based physical row indexes) of the rows the snapshot does
    /// not show, ascending and distinct.
    pub deleted_positions: Vec<i64>,
}

/// How a data file's physical columns map onto the table's `column_id`s.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColumnBinding {
    /// Every top-level column of the file carries a Parquet `field_id`, which is
    /// the `column_id` of the table column it holds.
    FieldIds,
    /// The file carries no field ids; the catalog maps its columns by name
    /// (`ducklake_name_mapping`), as for files added with
    /// `ducklake_add_data_files`.
    NameMapping(DuckLakeNameMapping),
    /// The file has columns without a field id and the catalog records no name
    /// mapping for it, so nothing says which table column they hold. Reported,
    /// not guessed: a target that matched them by name or position could bind
    /// the wrong column.
    Unresolved {
        /// The file's top-level columns that carry no field id.
        columns: Vec<String>,
    },
}

/// Why a table snapshot cannot be exported as a [`TableManifest`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// The snapshot holds rows inlined in the catalog, which no file carries.
    /// A manifest without them would be a wrong answer. Flush them
    /// (`DuckLakeCatalog::flush_inlined_data`, or
    /// `DuckLakeTableWriter::flush_table_inlined_data`) and export the flush
    /// snapshot.
    #[error(
        "table {table_id} has {rows} inlined row(s) at snapshot {snapshot_id}, which no data file \
         carries; flush the table's inlined data and export the flush snapshot"
    )]
    InlinedRows {
        table_id: i64,
        snapshot_id: i64,
        rows: usize,
    },
    /// A data file is encrypted with a key the catalog holds, so no other
    /// format can read it in place. Copy the table instead.
    #[error(
        "data file {path} is encrypted with a key held in the DuckLake catalog; another table \
         format cannot read it in place, so copy the data instead"
    )]
    EncryptedFile {
        path: String,
    },
}

impl DuckLakeTable {
    /// Export this table at the snapshot it was opened at as a
    /// [`TableManifest`].
    ///
    /// Reads the catalog's file listing, each delete file live at the snapshot,
    /// and the footer of each data file the catalog records no name mapping
    /// for (to check its field ids). Writes nothing.
    ///
    /// Fails with [`ManifestError::InlinedRows`] when the snapshot holds inlined
    /// rows, and with [`ManifestError::EncryptedFile`] when a data file is
    /// encrypted. Inlined *deletions* of Parquet rows are resolved into
    /// [`ManifestDataFile::deleted_positions`] like any other.
    pub async fn export_manifest(&self, state: &dyn Session) -> Result<TableManifest> {
        let inlined_rows: usize = self
            .inlined_data_with_row_ids()?
            .iter()
            .map(|inlined| inlined.batch.num_rows())
            .sum();
        if inlined_rows > 0 {
            return Err(ManifestError::InlinedRows {
                table_id: self.table_id(),
                snapshot_id: self.snapshot_id(),
                rows: inlined_rows,
            }
            .into());
        }

        let mut listed = Vec::new();
        for page in self.file_metadata_pages("manifest export", None) {
            listed.extend(page?);
        }
        for metadata in &listed {
            if metadata
                .file
                .file
                .encryption_key
                .as_deref()
                .is_some_and(|key| !key.is_empty())
            {
                return Err(ManifestError::EncryptedFile {
                    path: self.absolute_location(&metadata.file)?,
                }
                .into());
            }
        }

        let inlined_deletes = self.inlined_deletes_by_file()?;
        let mut mappings: HashMap<i64, DuckLakeNameMapping> = HashMap::new();
        let mut files = Vec::with_capacity(listed.len());
        for metadata in listed {
            let table_file = metadata.file;
            let path = self.absolute_location(&table_file)?;
            let mut deleted: HashSet<i64> = self
                .deleted_positions_at_snapshot(
                    state,
                    &table_file,
                    inlined_deletes.get(&table_file.data_file_id),
                )
                .await?;
            deleted.extend(self.positions_inserted_later(state, &table_file).await?);
            let mut deleted_positions: Vec<i64> = deleted.into_iter().collect();
            deleted_positions.sort_unstable();

            let (column_binding, footer_rows) = match table_file.file.mapping_id {
                Some(mapping_id) => {
                    let mapping = match mappings.get(&mapping_id) {
                        Some(mapping) => mapping.clone(),
                        None => {
                            let mapping = self.provider().get_name_mapping(mapping_id)?;
                            mappings.insert(mapping_id, mapping.clone());
                            mapping
                        },
                    };
                    (ColumnBinding::NameMapping(mapping), None)
                },
                None => {
                    let facts = read_parquet_footer_facts(
                        state,
                        self.object_store_url(),
                        &self.resolve_file_path(&table_file.file)?,
                        Some(table_file.file.file_size_bytes),
                        table_file.file.footer_size,
                        None,
                    )
                    .await?;
                    let columns: Vec<String> = facts
                        .arrow_schema
                        .fields()
                        .iter()
                        .filter(|field| !has_field_id(field))
                        .map(|field| field.name().clone())
                        .collect();
                    let binding = if columns.is_empty() {
                        ColumnBinding::FieldIds
                    } else {
                        ColumnBinding::Unresolved {
                            columns,
                        }
                    };
                    (binding, Some(facts.row_count))
                },
            };
            let record_count = match (table_file.max_row_count, footer_rows) {
                (Some(count), _) | (None, Some(count)) => count,
                (None, None) => self.data_file_row_count(state, &table_file.file).await?,
            };

            files.push(ManifestDataFile {
                data_file_id: table_file.data_file_id,
                path,
                file_size_bytes: table_file.file.file_size_bytes,
                record_count,
                footer_size: table_file.file.footer_size,
                column_statistics: metadata.column_statistics,
                partition_id: table_file.partition_id,
                partition_values: table_file.partition_values,
                column_binding,
                deleted_positions,
            });
        }

        Ok(TableManifest {
            table_id: self.table_id(),
            snapshot_id: self.snapshot_id(),
            fields: self.fields().to_vec(),
            partition_spec: self.partition_spec().cloned(),
            files,
        })
    }

    /// `table_file`'s location as an absolute URL.
    fn absolute_location(&self, table_file: &DuckLakeTableFile) -> Result<String> {
        let key = self.resolve_file_path(&table_file.file)?;
        let base = self.object_store_url().as_str();
        let separator = if base.ends_with('/') {
            ""
        } else {
            "/"
        };
        Ok(format!("{base}{separator}{}", key.trim_start_matches('/')))
    }
}

fn has_field_id(field: &Field) -> bool {
    field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY)
}
