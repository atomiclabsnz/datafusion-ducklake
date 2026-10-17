//! Import existing Parquet files, with the positions deleted in them, as one
//! new snapshot of a DuckLake table, without copying them.
//!
//! The input is a [`TableManifest`], the shape
//! [`DuckLakeTable::export_manifest`](crate::DuckLakeTable::export_manifest)
//! produces: the table's columns with their ids, and per file its location,
//! sizes, record count, statistics, partition values, column binding and
//! deleted positions. [`DuckLakeTableWriter::import_manifest`] is the Rust
//! counterpart of official DuckLake's `ducklake_add_data_files`, with deletes:
//!
//! - **Field ids must match.** A file's Parquet `field_id`s must be column ids of
//!   the table; a file without them is accepted only with a name mapping. Anything
//!   else is refused per file with an [`ImportError`] naming the file and column,
//!   never matched by position.
//! - The deleted positions become positional delete files committed in the
//!   **same** snapshot as the data files, so no snapshot shows the deleted rows.
//! - One snapshot for the whole import, through the writer's append conflict
//!   checks. An existing table whose columns differ from the manifest's is
//!   refused unless [`ImportOptions::allow_schema_evolution`] is set.
//! - **Ownership is explicit.** The files stay where they are, and by default the
//!   catalog records that it does not own them: cleanup never deletes them, even
//!   after expiry or compaction retires them. [`ImportOptions::take_ownership`]
//!   hands them to the catalog.
//! - Statistics come from the manifest when it has them for a file, else from the
//!   file's footer.
//!
//! A new table is created unpartitioned. Partition values are kept when the target
//! table's live partition spec has the manifest's keys; create the table and set
//! its spec first to keep them.
//!
//! The SQLite metadata writer implements the commit; other writers refuse it
//! with [`crate::DuckLakeError::Unsupported`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::{Field, SchemaRef};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};

use crate::manifest::{ColumnBinding, ManifestDataFile, TableManifest};
use crate::metadata_provider::{DuckLakeTableField, MetadataProvider};
use crate::metadata_writer::{CatalogNode, ColumnDef, ColumnStat, DataFileInfo, ImportedFile};
use crate::path_resolver::parse_object_store_url;
use crate::row_id::{ROW_ID_PARQUET_FIELD_ID, SNAPSHOT_ID_PARQUET_FIELD_ID};
use crate::table_writer::{DuckLakeTableWriter, is_transport_error, remove_objects};
use crate::types::build_arrow_schema_from_fields;
use crate::{DuckLakeError, Result};

/// How [`DuckLakeTableWriter::import_manifest`] treats the target table and the
/// imported objects.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ImportOptions {
    /// Hand the imported objects to the catalog, whose expiry, compaction and
    /// cleanup may then delete them. Off by default: the source table still
    /// reads them, so cleanup leaves them in place.
    pub take_ownership: bool,
    /// Let the import change an existing table's columns to the manifest's.
    /// Off by default: an existing table whose columns differ is refused.
    pub allow_schema_evolution: bool,
    /// The snapshot the caller checked the target against. A drop, alteration
    /// or delete of the table after it aborts the commit. Defaults to the
    /// catalog head when the import begins.
    pub base_snapshot: Option<i64>,
}

impl ImportOptions {
    /// Set [`Self::take_ownership`].
    pub fn with_take_ownership(mut self, take_ownership: bool) -> Self {
        self.take_ownership = take_ownership;
        self
    }

    /// Set [`Self::allow_schema_evolution`].
    pub fn with_allow_schema_evolution(mut self, allow: bool) -> Self {
        self.allow_schema_evolution = allow;
        self
    }

    /// Set [`Self::base_snapshot`].
    pub fn with_base_snapshot(mut self, snapshot_id: i64) -> Self {
        self.base_snapshot = Some(snapshot_id);
        self
    }
}

/// What an import committed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImportResult {
    /// The snapshot holding every imported file and its deletes.
    pub snapshot_id: i64,
    /// The target table.
    pub table_id: i64,
    /// Data files registered.
    pub files: usize,
    /// Rows the imported files hold, before deletes.
    pub record_count: i64,
    /// Positions written to delete files.
    pub deleted_rows: i64,
}

/// Why a manifest, or one of its files, cannot be imported.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ImportError {
    /// A column of the file carries a field id that is not a column id of the
    /// table, so it holds no column the table has under that id.
    #[error(
        "file {path}: column '{column}' has field id {field_id}, which is no column of the table"
    )]
    FieldIdMismatch {
        path: String,
        column: String,
        field_id: i64,
    },
    /// A column of the file carries no field id and the manifest gives no name
    /// mapping for the file.
    #[error("file {path}: column '{column}' has no field id and the file has no name mapping")]
    MissingFieldId {
        path: String,
        column: String,
    },
    /// A position is outside the file's rows, or listed twice.
    #[error(
        "file {path}: deleted position {position} is not a row of its {record_count} rows, or is listed twice"
    )]
    PositionOutOfRange {
        path: String,
        position: i64,
        record_count: i64,
    },
    /// The file's footer records a different number of rows than the manifest.
    #[error("file {path}: the manifest records {manifest} rows, its footer {footer}")]
    RecordCountMismatch {
        path: String,
        manifest: i64,
        footer: i64,
    },
    /// The file is listed twice, or is already a live file of the table.
    #[error("file {path} is listed twice, or is already a file of the table")]
    DuplicateFile {
        path: String,
    },
    /// The file is not in the object store the catalog's data path is in.
    #[error("file {path} is not in the catalog's object store {expected}")]
    ForeignObjectStore {
        path: String,
        expected: String,
    },
    /// The existing table's columns differ from the manifest's.
    #[error(
        "table {table} has columns {existing}, the manifest {manifest}; set allow_schema_evolution \
         to change them"
    )]
    SchemaMismatch {
        table: String,
        existing: String,
        manifest: String,
    },
}

/// One manifest file, checked and ready to register.
struct CheckedFile {
    info: DataFileInfo,
    name_mapping: Option<Vec<crate::metadata_provider::DuckLakeNameMappingEntry>>,
    deleted_positions: Vec<i64>,
}

impl DuckLakeTableWriter {
    /// Register `manifest`'s files, with their deleted positions, as one new
    /// snapshot of `schema_name.table_name`, creating the table with the
    /// manifest's columns and column ids when it does not exist. The files are
    /// not copied: each must be in the object store of the catalog's data path,
    /// which this writer's object store serves. See the [module
    /// docs](crate::manifest_import) for the rules.
    ///
    /// Reads each file's footer to check its row count and field ids (and, when
    /// the manifest has none for it, its statistics), writes one delete file per
    /// file with deleted positions, and commits. A refused file fails the import
    /// before anything is written; a failed commit removes the delete files.
    pub async fn import_manifest(
        &self,
        provider: Arc<dyn MetadataProvider>,
        schema_name: &str,
        table_name: &str,
        manifest: &TableManifest,
        options: &ImportOptions,
    ) -> Result<ImportResult> {
        let (columns, column_ids) = manifest_columns(&manifest.fields)?;
        let table_ids: HashSet<i64> = manifest
            .fields
            .iter()
            .map(|field| field.column_id)
            .collect();

        let head = provider.get_current_snapshot()?;
        if let Some(schema) = provider.get_schema_by_name(schema_name, head)?
            && let Some(table) = provider.get_table_by_name(schema.schema_id, table_name, head)?
        {
            let existing = provider.get_table_fields(table.table_id, head)?;
            if !options.allow_schema_evolution
                && field_signature(&existing) != field_signature(&manifest.fields)
            {
                return Err(ImportError::SchemaMismatch {
                    table: format!("{schema_name}.{table_name}"),
                    existing: describe_fields(&existing),
                    manifest: describe_fields(&manifest.fields),
                }
                .into());
            }
        }

        let (catalog_store, _) = parse_object_store_url(&self.metadata_writer().get_data_path()?)?;
        let mut seen = HashSet::new();
        let mut checked = Vec::with_capacity(manifest.files.len());
        for file in &manifest.files {
            if !seen.insert(file.path.as_str()) {
                return Err(ImportError::DuplicateFile {
                    path: file.path.clone(),
                }
                .into());
            }
            let (store, key) = parse_object_store_url(&file.path)?;
            if store != catalog_store {
                return Err(ImportError::ForeignObjectStore {
                    path: file.path.clone(),
                    expected: catalog_store.to_string(),
                }
                .into());
            }
            checked.push(self.check_file(file, key, &table_ids).await?);
        }

        let setup = self.metadata_writer().begin_write_transaction(
            schema_name,
            table_name,
            &columns,
            crate::metadata_writer::WriteMode::Append,
        )?;
        let base_snapshot = options.base_snapshot.unwrap_or(setup.base_snapshot_id);
        let live_spec = self.metadata_writer().live_partition_spec(setup.table_id)?;
        let keep_partitions = match (&live_spec, &manifest.partition_spec) {
            (Some(live), Some(exported)) => {
                let keys = |spec: &crate::partition::PartitionSpec| {
                    spec.columns
                        .iter()
                        .map(|c| (c.partition_key_index, c.column_id, c.transform.clone()))
                        .collect::<Vec<_>>()
                };
                (keys(live) == keys(exported)).then_some((live.partition_id, exported.partition_id))
            },
            _ => None,
        };

        let mut written: Vec<ObjectPath> = Vec::new();
        let mut imported = Vec::with_capacity(checked.len());
        let mut record_count = 0;
        let mut deleted_rows = 0;
        let result = async {
            for (file, manifest_file) in checked.into_iter().zip(&manifest.files) {
                let mut info = file.info;
                if let Some((live_id, exported_id)) = keep_partitions
                    && manifest_file.partition_id == Some(exported_id)
                {
                    info = info.with_partition(live_id, manifest_file.partition_values.clone());
                }
                record_count += info.record_count;
                let mut entry = ImportedFile::new(info.clone()).with_owned(options.take_ownership);
                if let Some(mapping) = file.name_mapping {
                    entry = entry.with_name_mapping(mapping);
                }
                if !file.deleted_positions.is_empty() {
                    let delete = self
                        .write_delete_file(
                            schema_name,
                            table_name,
                            &info.path,
                            &file.deleted_positions,
                        )
                        .await?;
                    written.push(self.delete_file_object_path(
                        schema_name,
                        table_name,
                        &delete.path,
                    )?);
                    deleted_rows += delete.delete_count;
                    entry = entry.with_delete(delete);
                }
                imported.push(entry);
            }
            self.metadata_writer().register_imported_files(
                setup.table_id,
                schema_name,
                table_name,
                &imported,
                base_snapshot,
                &columns,
                &column_ids,
            )
        }
        .await;
        match result {
            Ok(commit) => Ok(ImportResult {
                snapshot_id: commit.snapshot_id,
                table_id: commit.table_id,
                files: imported.len(),
                record_count,
                deleted_rows,
            }),
            Err(error) => {
                if !is_transport_error(&error) {
                    remove_objects(self.object_store(), written).await;
                }
                Err(error)
            },
        }
    }

    /// Check one manifest file against its footer and the table's column ids.
    async fn check_file(
        &self,
        file: &ManifestDataFile,
        key: String,
        table_ids: &HashSet<i64>,
    ) -> Result<CheckedFile> {
        let (metadata, footer_size) =
            read_footer(self.object_store(), &key, file.file_size_bytes).await?;
        let footer_rows = metadata.file_metadata().num_rows();
        if footer_rows != file.record_count {
            return Err(ImportError::RecordCountMismatch {
                path: file.path.clone(),
                manifest: file.record_count,
                footer: footer_rows,
            }
            .into());
        }
        let metadata = Arc::new(metadata);
        let arrow_schema: SchemaRef =
            ArrowReaderMetadata::try_new(Arc::clone(&metadata), ArrowReaderOptions::new())?
                .schema()
                .clone();

        // The column id each top-level column of the file holds, if any.
        let (file_column_ids, name_mapping) = match &file.column_binding {
            ColumnBinding::FieldIds => {
                let mut ids = Vec::with_capacity(arrow_schema.fields().len());
                for field in arrow_schema.fields() {
                    let Some(id) = field_id(field) else {
                        return Err(ImportError::MissingFieldId {
                            path: file.path.clone(),
                            column: field.name().clone(),
                        }
                        .into());
                    };
                    if id == i64::from(ROW_ID_PARQUET_FIELD_ID)
                        || id == i64::from(SNAPSHOT_ID_PARQUET_FIELD_ID)
                    {
                        ids.push(None);
                        continue;
                    }
                    if !table_ids.contains(&id) {
                        return Err(ImportError::FieldIdMismatch {
                            path: file.path.clone(),
                            column: field.name().clone(),
                            field_id: id,
                        }
                        .into());
                    }
                    ids.push(Some(id));
                }
                (ids, None)
            },
            ColumnBinding::NameMapping(mapping) => {
                for entry in &mapping.entries {
                    if !table_ids.contains(&entry.target_field_id) {
                        return Err(ImportError::FieldIdMismatch {
                            path: file.path.clone(),
                            column: entry.source_name.clone(),
                            field_id: entry.target_field_id,
                        }
                        .into());
                    }
                }
                let by_name: HashMap<&str, i64> = mapping
                    .entries
                    .iter()
                    .filter(|entry| entry.parent_column.is_none())
                    .map(|entry| (entry.source_name.as_str(), entry.target_field_id))
                    .collect();
                let ids = arrow_schema
                    .fields()
                    .iter()
                    .map(|field| by_name.get(field.name().as_str()).copied())
                    .collect();
                (ids, Some(mapping.entries.clone()))
            },
            ColumnBinding::Unresolved {
                columns,
            } => {
                return Err(ImportError::MissingFieldId {
                    path: file.path.clone(),
                    column: columns.first().cloned().unwrap_or_default(),
                }
                .into());
            },
        };

        let mut previous = -1;
        for &position in &file.deleted_positions {
            if position <= previous || position >= file.record_count {
                return Err(ImportError::PositionOutOfRange {
                    path: file.path.clone(),
                    position,
                    record_count: file.record_count,
                }
                .into());
            }
            previous = position;
        }

        let column_stats: Vec<ColumnStat> = if file.column_statistics.is_empty() {
            // `compute_column_stats` labels file column `i` with `ids[i]`; a
            // column the table does not hold gets an id no stat is kept for.
            let labels: Vec<i64> = file_column_ids.iter().map(|id| id.unwrap_or(-1)).collect();
            crate::stats_collect::compute_column_stats(
                &arrow_schema,
                &metadata,
                &labels,
                footer_rows,
                &[],
            )
        } else {
            file.column_statistics
                .iter()
                .map(|stat| ColumnStat {
                    column_id: stat.column_id,
                    min_value: stat.min_value.clone(),
                    max_value: stat.max_value.clone(),
                    null_count: stat.null_count,
                    value_count: stat.value_count,
                    contains_nan: stat.contains_nan,
                    column_size_bytes: stat.column_size_bytes,
                })
                .collect()
        };
        let mut stat_ids = HashSet::new();
        let column_stats = column_stats
            .into_iter()
            .filter(|stat| table_ids.contains(&stat.column_id) && stat_ids.insert(stat.column_id))
            .collect();

        let mut info = DataFileInfo::new(key, file.file_size_bytes, file.record_count)
            .with_footer_size(file.footer_size.unwrap_or(footer_size))
            .with_column_stats(column_stats);
        info.path_is_relative = false;
        Ok(CheckedFile {
            info,
            name_mapping,
            deleted_positions: file.deleted_positions.clone(),
        })
    }
}

/// The footer of the Parquet object `key`, `size` bytes long, and the length
/// of its metadata (DuckLake's `footer_size`).
async fn read_footer(
    store: &Arc<dyn ObjectStore>,
    key: &str,
    size: i64,
) -> Result<(ParquetMetaData, i64)> {
    let path = ObjectPath::from(key.trim_start_matches('/'));
    let size = u64::try_from(size).map_err(|_| {
        DuckLakeError::InvalidConfig(format!("file {key} has a negative size {size}"))
    })?;
    if size < 12 {
        return Err(DuckLakeError::InvalidConfig(format!(
            "file {key} is too small to be Parquet"
        )));
    }
    let tail = store.get_range(&path, size - 8..size).await?;
    if &tail[4..8] != b"PAR1" {
        return Err(DuckLakeError::InvalidConfig(format!(
            "file {key} is not Parquet (or is encrypted)"
        )));
    }
    let metadata_len = u64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
    if metadata_len + 8 > size {
        return Err(DuckLakeError::InvalidConfig(format!(
            "file {key} records a footer longer than the file"
        )));
    }
    let footer = store
        .get_range(&path, size - 8 - metadata_len..size - 8)
        .await?;
    let metadata = ParquetMetaDataReader::decode_metadata(&footer)?;
    Ok((metadata, metadata_len as i64))
}

fn field_id(field: &Field) -> Option<i64> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)?
        .parse()
        .ok()
}

/// The manifest's columns as the writer creates them, and their ids depth
/// first, as `begin_write_transaction` and the commit take them.
fn manifest_columns(fields: &[DuckLakeTableField]) -> Result<(Vec<ColumnDef>, Vec<i64>)> {
    let schema = build_arrow_schema_from_fields(fields)?;
    let roots: Vec<&DuckLakeTableField> = fields
        .iter()
        .filter(|field| field.parent_column.is_none())
        .collect();
    if roots.len() != schema.fields().len() {
        return Err(DuckLakeError::Internal(format!(
            "the manifest has {} top-level columns for {} schema fields",
            roots.len(),
            schema.fields().len()
        )));
    }
    let mut columns = Vec::with_capacity(roots.len());
    let mut ids = Vec::with_capacity(fields.len());
    for (arrow_field, root) in schema.fields().iter().zip(roots) {
        let mut nodes = Vec::new();
        let mut stack = vec![root];
        while let Some(field) = stack.pop() {
            ids.push(field.column_id);
            nodes.push(CatalogNode {
                ducklake_type: field.column_type.clone(),
                is_nullable: field.is_nullable,
                initial_default: None,
                default_value: None,
                default_value_type: None,
                default_value_dialect: None,
            });
            let children: Vec<&DuckLakeTableField> = fields
                .iter()
                .filter(|child| child.parent_column == Some(field.column_id))
                .collect();
            stack.extend(children.into_iter().rev());
        }
        columns.push(ColumnDef::copying(
            arrow_field.name(),
            arrow_field.data_type(),
            nodes,
        )?);
    }
    Ok((columns, ids))
}

/// What a schema check compares: each column's id, name, type and parent.
fn field_signature(fields: &[DuckLakeTableField]) -> Vec<(i64, String, String, Option<i64>)> {
    let mut signature: Vec<_> = fields
        .iter()
        .map(|field| {
            (
                field.column_id,
                field.column_name.clone(),
                field.column_type.to_ascii_lowercase(),
                field.parent_column,
            )
        })
        .collect();
    signature.sort();
    signature
}

fn describe_fields(fields: &[DuckLakeTableField]) -> String {
    let columns: Vec<String> = fields
        .iter()
        .filter(|field| field.parent_column.is_none())
        .map(|field| {
            format!(
                "{} {} (id {})",
                field.column_name, field.column_type, field.column_id
            )
        })
        .collect();
    format!("[{}]", columns.join(", "))
}
