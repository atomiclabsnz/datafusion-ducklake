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
//!   the table, or of a column it has dropped
//!   ([`TableManifest::retired_fields`]); a file without them is accepted only
//!   with a name mapping. Anything else is refused per file with an
//!   [`ImportError`] naming the file and column, never matched by position.
//! - **A dropped column's id stays taken.** Each retired field is recorded as a
//!   column that is never visible (it begins and ends at the import's snapshot),
//!   so neither DuckDB nor this crate gives its id to a column added later, which
//!   would then read the dropped column's values out of the older files.
//! - The deleted positions become positional delete files committed in the
//!   **same** snapshot as the data files, so no snapshot shows the deleted rows.
//! - One snapshot for the whole import, through the writer's append conflict
//!   checks. An existing table whose columns differ from the manifest's is
//!   refused unless [`ImportOptions::allow_schema_evolution`] is set.
//! - **Ownership is explicit.** The files stay where they are, and by default the
//!   catalog records that it does not own them: cleanup never deletes them, even
//!   after expiry or compaction retires them. [`ImportOptions::take_ownership`]
//!   hands them to the catalog.
//! - **Statistics come from the file's footer, never from the manifest.** A
//!   bound is kept only when every row group of the file records an exact
//!   minimum and maximum, and the counts only when every row group records its
//!   null count: a statistic the import cannot verify is not written, since a
//!   reader prunes files and answers `min`/`max` by it. The footer length stored
//!   is the one measured.
//! - **Each file column is held to the table column of its id**: the same type,
//!   or a promotion DuckDB reads (a wider integer, `float` to `double`, a
//!   decimal's precision at the same scale). A `NOT NULL` column must be in the
//!   file with a recorded null count of 0. Nested field ids must be columns of
//!   the table or retired fields, as top-level ones must.
//! - **Partition values are verified** before they are kept: an `identity`
//!   value must equal the file's exact minimum and maximum of the column (NULL:
//!   every value NULL), a `year` value the calendar year of both. Any other
//!   transform's values cannot be verified from a footer and are refused.
//!
//! A new table is created unpartitioned. Partition values are kept when the target
//! table's live partition spec has the manifest's keys; create the table and set
//! its spec first to keep them.
//!
//! The SQLite metadata writer implements the commit; other writers refuse it
//! with [`crate::DuckLakeError::Unsupported`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::{Field, SchemaRef};
use datafusion::common::ScalarValue;
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
    /// A retired field cannot be one: its id is not a positive non-reserved id,
    /// it is listed twice, or its parent is no column of the table or retired
    /// field.
    #[error("retired field '{column}' (id {field_id}): {reason}")]
    InvalidRetiredField {
        column: String,
        field_id: i64,
        reason: String,
    },
    /// A file column's type is not the table column's, nor a promotion DuckDB
    /// reads into it.
    #[error(
        "file {path}: column '{column}' is {file_type} in the file, which DuckDB does not read \
         as the table's {table_type}"
    )]
    ColumnTypeMismatch {
        path: String,
        column: String,
        file_type: String,
        table_type: String,
    },
    /// A `NOT NULL` column has NULLs in the file, is missing from it, or the
    /// file records no null count for it.
    #[error("file {path}: NOT NULL column '{column}' {reason}")]
    NotNullViolated {
        path: String,
        column: String,
        reason: String,
    },
    /// A partition value the file's footer does not show to be the file's.
    #[error("file {path}: partition value of column {column_id} cannot be verified: {reason}")]
    UnverifiedPartitionValue {
        path: String,
        column_id: i64,
        reason: String,
    },
    /// A retired field has the id of one of the table's columns.
    #[error("retired field '{column}' has id {field_id}, which a column of the table has")]
    RetiredFieldIsLive {
        column: String,
        field_id: i64,
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
    /// What the footer shows of each top-level scalar column, by column id.
    facts: HashMap<i64, ColumnFacts>,
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
        let data_path = self.metadata_writer().get_data_path()?;
        let Prepared {
            columns,
            column_ids,
            checked,
        } = prepare(
            self.object_store(),
            provider.as_ref(),
            &data_path,
            schema_name,
            table_name,
            manifest,
            options,
        )
        .await?;

        let setup = self.metadata_writer().begin_write_transaction(
            schema_name,
            table_name,
            &columns,
            crate::metadata_writer::WriteMode::Append,
        )?;
        let release = crate::table_writer::ReleaseOnFailure::new(self.metadata_writer(), &setup);
        let base_snapshot = options.base_snapshot.unwrap_or(setup.base_snapshot_id);
        let live_spec = self.metadata_writer().live_partition_spec(setup.table_id)?;
        let keep_partitions = match (&live_spec, &manifest.partition_spec) {
            (Some(live), Some(exported)) if partition_keys(live) == partition_keys(exported) => {
                Some((live.partition_id, exported.partition_id))
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
                    if let Some(spec) = &live_spec {
                        verify_partition_values(
                            &manifest_file.path,
                            spec,
                            &manifest_file.partition_values,
                            &file.facts,
                        )?;
                    }
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
                &manifest.retired_fields,
            )
        }
        .await;
        match release.settle(result) {
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
}

/// The manifest's columns and its files, every check an import makes before it
/// writes having passed.
struct Prepared {
    columns: Vec<ColumnDef>,
    column_ids: Vec<i64>,
    checked: Vec<CheckedFile>,
}

/// Check `manifest` against the catalog `provider` reads and the files in
/// `store` (the object store of the catalog's `data_path`), exactly as
/// [`DuckLakeTableWriter::import_manifest`] does before it writes anything,
/// without a writer: so a caller can refuse a manifest before it opens one (a
/// writer on a catalog DuckDB created sets up its own tables and counters).
///
/// Reads the catalog and each file's footer; writes nothing. An import that
/// follows can still fail on a concurrent change (another writer creating the
/// table or committing first), as any write can.
pub async fn validate_manifest(
    store: &Arc<dyn ObjectStore>,
    provider: &dyn MetadataProvider,
    schema_name: &str,
    table_name: &str,
    manifest: &TableManifest,
    options: &ImportOptions,
) -> Result<()> {
    let data_path = provider.get_data_path()?;
    prepare(
        store,
        provider,
        &data_path,
        schema_name,
        table_name,
        manifest,
        options,
    )
    .await
    .map(|_| ())
}

/// Every check of an import that precedes its first write: the shared body of
/// [`validate_manifest`] and [`DuckLakeTableWriter::import_manifest`], so the two
/// cannot disagree.
async fn prepare(
    store: &Arc<dyn ObjectStore>,
    provider: &dyn MetadataProvider,
    data_path: &str,
    schema_name: &str,
    table_name: &str,
    manifest: &TableManifest,
    options: &ImportOptions,
) -> Result<Prepared> {
    let (columns, column_ids) = manifest_columns(&manifest.fields)?;
    let table_ids: HashSet<i64> = manifest
        .fields
        .iter()
        .map(|field| field.column_id)
        .collect();
    if let Some(live) = manifest
        .retired_fields
        .iter()
        .find(|field| table_ids.contains(&field.column_id))
    {
        return Err(ImportError::RetiredFieldIsLive {
            column: live.column_name.clone(),
            field_id: live.column_id,
        }
        .into());
    }
    let retired_ids: HashSet<i64> = manifest
        .retired_fields
        .iter()
        .map(|field| field.column_id)
        .collect();
    validate_retired(&manifest.retired_fields, &table_ids, &retired_ids)?;

    let head = provider.get_current_snapshot()?;
    let mut live_spec = None;
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
        live_spec = provider.get_partition_spec(table.table_id, head)?;
    }
    let kept_spec = match (&live_spec, &manifest.partition_spec) {
        (Some(live), Some(exported)) if partition_keys(live) == partition_keys(exported) => {
            Some((live.clone(), exported.partition_id))
        },
        _ => None,
    };

    let (catalog_store, _) = parse_object_store_url(data_path)?;
    let mut seen = HashSet::new();
    let mut checked = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        if !seen.insert(file.path.as_str()) {
            return Err(ImportError::DuplicateFile {
                path: file.path.clone(),
            }
            .into());
        }
        let (file_store, key) = parse_object_store_url(&file.path)?;
        if file_store != catalog_store {
            return Err(ImportError::ForeignObjectStore {
                path: file.path.clone(),
                expected: catalog_store.to_string(),
            }
            .into());
        }
        let file_checked =
            check_file(store, file, key, &manifest.fields, &table_ids, &retired_ids).await?;
        if let Some((spec, exported_id)) = &kept_spec
            && file.partition_id == Some(*exported_id)
        {
            verify_partition_values(
                &file.path,
                spec,
                &file.partition_values,
                &file_checked.facts,
            )?;
        }
        checked.push(file_checked);
    }
    Ok(Prepared {
        columns,
        column_ids,
        checked,
    })
}

/// A partition spec's keys, as two specs are compared to decide whether a
/// file's values carry over.
fn partition_keys(
    spec: &crate::partition::PartitionSpec,
) -> Vec<(i32, i64, crate::partition::PartitionTransform)> {
    spec.columns
        .iter()
        .map(|c| (c.partition_key_index, c.column_id, c.transform.clone()))
        .collect()
}

/// Check one manifest file against its footer and the table's column ids.
async fn check_file(
    store: &Arc<dyn ObjectStore>,
    file: &ManifestDataFile,
    key: String,
    fields: &[DuckLakeTableField],
    table_ids: &HashSet<i64>,
    retired_ids: &HashSet<i64>,
) -> Result<CheckedFile> {
    let (metadata, footer_size) = read_footer(store, &key, file.file_size_bytes).await?;
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
                // A dropped column's values, which the table no longer shows.
                if id == i64::from(ROW_ID_PARQUET_FIELD_ID)
                    || id == i64::from(SNAPSHOT_ID_PARQUET_FIELD_ID)
                    || retired_ids.contains(&id)
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
            // A nested field id must name a column too: a dropped struct member's
            // id left unreserved could be given to a member added later.
            for field in arrow_schema.fields() {
                let mut nested = Vec::new();
                nested_ids(field, &mut nested);
                if let Some((column, id)) = nested
                    .into_iter()
                    .find(|(_, id)| !table_ids.contains(id) && !retired_ids.contains(id))
                {
                    return Err(ImportError::FieldIdMismatch {
                        path: file.path.clone(),
                        column,
                        field_id: id,
                    }
                    .into());
                }
            }
            (ids, None)
        },
        ColumnBinding::NameMapping(mapping) => {
            for entry in &mapping.entries {
                if !table_ids.contains(&entry.target_field_id)
                    && !retired_ids.contains(&entry.target_field_id)
                {
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
                // A Hive-path column is read from the path, never from the file: a file column
                // of the same name binds nothing, so no statistic of it is written (a scan
                // would prune the path's values by the file's).
                .filter(|entry| {
                    entry.parent_column.is_none()
                        && !entry.is_partition
                        && table_ids.contains(&entry.target_field_id)
                })
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

    let leaf_of = leaf_of_id(&metadata, &file.column_binding, &file_column_ids);
    // The columns a name mapping reads from the file's Hive path (`is_partition`), by
    // target column id: a file never holds them, so a `NOT NULL` one is held to its path
    // value instead of to a null count.
    let hive_columns: HashMap<i64, &str> = match &file.column_binding {
        ColumnBinding::NameMapping(mapping) => mapping
            .entries
            .iter()
            .filter(|entry| entry.is_partition && entry.parent_column.is_none())
            .map(|entry| (entry.target_field_id, entry.source_name.as_str()))
            .collect(),
        _ => HashMap::new(),
    };
    check_types_and_nulls(
        &file.path,
        fields,
        &arrow_schema,
        &file_column_ids,
        &leaf_of,
        &metadata,
        &hive_columns,
    )?;
    let facts = column_facts(&arrow_schema, &metadata, &file_column_ids);
    let exact_types = bounds_carry_over(fields, &arrow_schema, &file_column_ids)?;

    // Always the footer's (see the module docs), and only what every row
    // group records exactly. `compute_column_stats` labels file column `i`
    // with `ids[i]`; a column the table does not hold gets an id no stat is
    // kept for.
    let labels: Vec<i64> = file_column_ids.iter().map(|id| id.unwrap_or(-1)).collect();
    let column_stats: Vec<ColumnStat> = crate::stats_collect::compute_column_stats(
        &arrow_schema,
        &metadata,
        &labels,
        footer_rows,
        &[],
    )
    .into_iter()
    .map(|mut stat| {
        let known = facts.get(&stat.column_id);
        // A bound is the file's value: written only when the table's type reads
        // that value as itself (a float32 0.1 is not the float64 0.1).
        if !known.is_some_and(|f| f.exact_bounds) || !exact_types.contains(&stat.column_id) {
            stat.min_value = None;
            stat.max_value = None;
        }
        if known.is_none_or(|f| f.null_count.is_none()) {
            stat.null_count = None;
            stat.value_count = None;
        }
        stat
    })
    .collect();
    let mut stat_ids = HashSet::new();
    let column_stats = column_stats
        .into_iter()
        .filter(|stat| table_ids.contains(&stat.column_id) && stat_ids.insert(stat.column_id))
        .collect();

    let mut info = DataFileInfo::new(key, file.file_size_bytes, file.record_count)
        // The length measured, whatever the manifest says: DuckDB refuses a
        // file whose stored footer length differs from the file's own.
        .with_footer_size(footer_size)
        .with_column_stats(column_stats);
    info.path_is_relative = false;
    Ok(CheckedFile {
        info,
        facts,
        name_mapping,
        deleted_positions: file.deleted_positions.clone(),
    })
}

/// The first id at which Iceberg and DuckLake reserve field ids for their own
/// metadata columns.
const FIRST_RESERVED_FIELD_ID: i64 = 2_147_483_447;

/// Refuse a retired field that cannot be one: an id that is not positive, is in
/// the reserved range, or is listed twice; or a parent that is no column of the
/// table or retired field.
fn validate_retired(
    retired: &[DuckLakeTableField],
    table_ids: &HashSet<i64>,
    retired_ids: &HashSet<i64>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for field in retired {
        let refuse = |reason: &str| -> Result<()> {
            Err(ImportError::InvalidRetiredField {
                column: field.column_name.clone(),
                field_id: field.column_id,
                reason: reason.to_string(),
            }
            .into())
        };
        if field.column_id <= 0 {
            return refuse("a column id is positive");
        }
        if field.column_id >= FIRST_RESERVED_FIELD_ID {
            return refuse("the id is in the range reserved for metadata columns");
        }
        if !seen.insert(field.column_id) {
            return refuse("it is listed twice");
        }
        if let Some(parent) = field.parent_column
            && !table_ids.contains(&parent)
            && !retired_ids.contains(&parent)
        {
            return refuse("its parent is no column of the table and no retired field");
        }
    }
    Ok(())
}

/// Every field nested under `field` (not `field` itself) that carries a field
/// id, with its name.
fn nested_ids(field: &Field, out: &mut Vec<(String, i64)>) {
    use arrow::datatypes::DataType;
    let children: Vec<&Field> = match field.data_type() {
        DataType::Struct(children) => children.iter().map(|c| c.as_ref()).collect(),
        DataType::List(child)
        | DataType::LargeList(child)
        | DataType::ListView(child)
        | DataType::LargeListView(child)
        | DataType::FixedSizeList(child, _)
        | DataType::Map(child, _) => vec![child.as_ref()],
        _ => Vec::new(),
    };
    for child in children {
        // A map's entries struct carries no id of its own.
        if let Some(id) = self::field_id(child) {
            out.push((child.name().clone(), id));
        }
        nested_ids(child, out);
    }
}

/// The Parquet leaf index of each field id the file binds: by the leaves' own
/// field ids, or, for a name-mapped file, by its top-level columns' mapped ids.
fn leaf_of_id(
    metadata: &ParquetMetaData,
    binding: &ColumnBinding,
    file_column_ids: &[Option<i64>],
) -> HashMap<i64, usize> {
    let descr = metadata.file_metadata().schema_descr();
    let mut out = HashMap::new();
    for leaf in 0..descr.num_columns() {
        let info = descr.column(leaf).self_type().get_basic_info().clone();
        match binding {
            ColumnBinding::NameMapping(_) => {
                // Only a top-level scalar column is its own leaf.
                let root = descr.get_column_root_idx(leaf);
                if descr.column(leaf).path().parts().len() == 1
                    && let Some(Some(id)) = file_column_ids.get(root)
                {
                    out.insert(*id, leaf);
                }
            },
            _ => {
                if info.has_id() {
                    out.insert(i64::from(info.id()), leaf);
                }
            },
        }
    }
    out
}

/// The file's Arrow type of each field id it binds.
fn file_types(
    arrow_schema: &SchemaRef,
    file_column_ids: &[Option<i64>],
) -> HashMap<i64, arrow::datatypes::DataType> {
    fn walk(field: &Field, out: &mut HashMap<i64, arrow::datatypes::DataType>) {
        if let Some(id) = self::field_id(field) {
            out.insert(id, field.data_type().clone());
        }
        use arrow::datatypes::DataType;
        match field.data_type() {
            DataType::Struct(children) => children.iter().for_each(|c| walk(c, out)),
            DataType::List(child)
            | DataType::LargeList(child)
            | DataType::FixedSizeList(child, _)
            | DataType::Map(child, _) => walk(child, out),
            _ => {},
        }
    }
    let mut out = HashMap::new();
    for (field, id) in arrow_schema.fields().iter().zip(file_column_ids) {
        match id {
            // A name-mapped column takes its id from the mapping.
            Some(id) if self::field_id(field).is_none() => {
                out.insert(*id, field.data_type().clone());
            },
            _ => walk(field, &mut out),
        }
    }
    out
}

/// The decimal precision and scale of a decimal type.
fn decimal_of(data_type: &arrow::datatypes::DataType) -> Option<(u8, i8)> {
    use arrow::datatypes::DataType;
    match data_type {
        DataType::Decimal32(p, s)
        | DataType::Decimal64(p, s)
        | DataType::Decimal128(p, s)
        | DataType::Decimal256(p, s) => Some((*p, *s)),
        _ => None,
    }
}

/// The column ids whose file values the table's type reads as the same values,
/// so a bound taken from the file is the table's: see [`bound_carries`].
fn bounds_carry_over(
    fields: &[DuckLakeTableField],
    arrow_schema: &SchemaRef,
    file_column_ids: &[Option<i64>],
) -> Result<HashSet<i64>> {
    let held = file_types(arrow_schema, file_column_ids);
    let mut out = HashSet::new();
    for field in fields {
        if let Some(file_type) = held.get(&field.column_id)
            && !matches!(
                field.column_type.to_ascii_lowercase().as_str(),
                "struct" | "list" | "map"
            )
            && bound_carries(
                file_type,
                &crate::types::ducklake_to_arrow_type(&field.column_type)?,
            )
        {
            out.insert(field.column_id);
        }
    }
    Ok(out)
}

/// Whether a file's minimum or maximum, of type `file`, is the same value read
/// as the table's `table`: the same type (any string or binary encoding of it,
/// any time zone label of a zoned instant), a wider integer, or a decimal's
/// precision widened at the same scale. Never a float read as a wider float:
/// a `float` bound printed as `0.1` is not the `double` the column holds.
fn bound_carries(file: &arrow::datatypes::DataType, table: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType::*;
    match (file, table) {
        (Float16 | Float32 | Float64, _) | (_, Float16 | Float32 | Float64) => file == table,
        _ => reads_as(file, table),
    }
}

/// Whether DuckDB reads a file column of type `file` as a table column of type
/// `table`: the same type (any string or binary encoding of it, any time zone
/// of a zoned timestamp), a wider integer of the same signedness or a wider
/// signed one for an unsigned, `float` to `double`, or a decimal's precision
/// widened at the same scale. Nested types are compared by their leaves.
fn reads_as(file: &arrow::datatypes::DataType, table: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType::*;
    let signed = |t: &arrow::datatypes::DataType| match t {
        Int8 => Some(8),
        Int16 => Some(16),
        Int32 => Some(32),
        Int64 => Some(64),
        _ => None,
    };
    let unsigned = |t: &arrow::datatypes::DataType| match t {
        UInt8 => Some(8),
        UInt16 => Some(16),
        UInt32 => Some(32),
        UInt64 => Some(64),
        _ => None,
    };
    if file == table {
        return true;
    }
    match (file, table) {
        (Utf8 | LargeUtf8 | Utf8View, Utf8 | LargeUtf8 | Utf8View) => true,
        (Binary | LargeBinary | BinaryView, Binary | LargeBinary | BinaryView) => true,
        (Timestamp(a, za), Timestamp(b, zb)) => a == b && za.is_some() == zb.is_some(),
        (Float32, Float64) => true,
        (Struct(_), Struct(_))
        | (List(_) | LargeList(_), List(_) | LargeList(_))
        | (Map(..), Map(..)) => true,
        _ => {
            if let (Some(a), Some(b)) = (signed(file), signed(table)) {
                return a <= b;
            }
            if let (Some(a), Some(b)) = (unsigned(file), unsigned(table)) {
                return a <= b;
            }
            if let (Some(a), Some(b)) = (unsigned(file), signed(table)) {
                return a < b;
            }
            if let (Some((fp, fs)), Some((tp, ts))) = (decimal_of(file), decimal_of(table)) {
                return fs == ts && fp <= tp;
            }
            false
        },
    }
}

/// Hold each file column to the table column of its id (see [`reads_as`]), and
/// each `NOT NULL` top-level column to a recorded null count of 0.
fn check_types_and_nulls(
    path: &str,
    fields: &[DuckLakeTableField],
    arrow_schema: &SchemaRef,
    file_column_ids: &[Option<i64>],
    leaf_of: &HashMap<i64, usize>,
    metadata: &ParquetMetaData,
    hive_columns: &HashMap<i64, &str>,
) -> Result<()> {
    let held = file_types(arrow_schema, file_column_ids);
    for field in fields {
        let nested = matches!(
            field.column_type.to_ascii_lowercase().as_str(),
            "struct" | "list" | "map"
        );
        if nested && let Some(file_type) = held.get(&field.column_id) {
            use arrow::datatypes::DataType;
            let kind = field.column_type.to_ascii_lowercase();
            let same_kind = match kind.as_str() {
                "struct" => matches!(file_type, DataType::Struct(_)),
                "list" => matches!(
                    file_type,
                    DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
                ),
                _ => matches!(file_type, DataType::Map(..)),
            };
            if !same_kind {
                return Err(ImportError::ColumnTypeMismatch {
                    path: path.to_string(),
                    column: field.column_name.clone(),
                    file_type: file_type.to_string(),
                    table_type: field.column_type.clone(),
                }
                .into());
            }
        }
        if !nested && let Some(file_type) = held.get(&field.column_id) {
            let table_type = crate::types::ducklake_to_arrow_type(&field.column_type)?;
            if !reads_as(file_type, &table_type) {
                return Err(ImportError::ColumnTypeMismatch {
                    path: path.to_string(),
                    column: field.column_name.clone(),
                    file_type: file_type.to_string(),
                    table_type: field.column_type.clone(),
                }
                .into());
            }
        }
        if field.is_nullable || field.parent_column.is_some() {
            continue;
        }
        let violated = |reason: &str| -> Result<()> {
            Err(ImportError::NotNullViolated {
                path: path.to_string(),
                column: field.column_name.clone(),
                reason: reason.to_string(),
            }
            .into())
        };
        if let Some(source_name) = hive_columns.get(&field.column_id) {
            // Read from the path exactly as a scan reads it: NULL spellings, decoding and the
            // column's type included.
            let table_type = crate::types::ducklake_to_arrow_type(&field.column_type)?;
            match crate::table::DuckLakeTable::hive_partition_value(path, source_name, &table_type)
            {
                Ok(value) if !value.is_null() => continue,
                Ok(_) => return violated("reads NULL from the file's Hive path"),
                Err(_) => {
                    return violated("has no value of its type in the file's Hive path");
                },
            }
        }
        if !held.contains_key(&field.column_id) {
            return violated("is not in the file, which reads it as NULL");
        }
        if nested {
            continue;
        }
        let Some(leaf) = leaf_of.get(&field.column_id) else {
            return violated("has no column chunk the null count can be read from");
        };
        match null_count(metadata, *leaf) {
            Some(0) => {},
            Some(_) => return violated("holds NULLs in the file"),
            None => return violated("has no null count recorded in the file"),
        }
    }
    Ok(())
}

/// The null count of Parquet leaf `leaf` across the file, when every row group
/// records it.
fn null_count(metadata: &ParquetMetaData, leaf: usize) -> Option<u64> {
    metadata
        .row_groups()
        .iter()
        .map(|rg| {
            rg.column(leaf)
                .statistics()
                .and_then(|s| s.null_count_opt())
        })
        .sum()
}

/// What a file's footer shows of one top-level scalar column.
#[derive(Debug, Clone)]
struct ColumnFacts {
    /// Every row group records an exact minimum and maximum.
    exact_bounds: bool,
    /// The file's minimum and maximum, when `exact_bounds` (None if all NULL).
    bounds: Option<(ScalarValue, ScalarValue)>,
    /// The null count, when every row group records it.
    null_count: Option<u64>,
    /// The file's rows.
    rows: u64,
}

/// [`ColumnFacts`] of each top-level scalar column the file binds, by column id.
fn column_facts(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    file_column_ids: &[Option<i64>],
) -> HashMap<i64, ColumnFacts> {
    use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
    let descr = metadata.file_metadata().schema_descr();
    let rows = u64::try_from(metadata.file_metadata().num_rows()).unwrap_or(0);
    let mut out = HashMap::new();
    for (root, (field, id)) in arrow_schema
        .fields()
        .iter()
        .zip(file_column_ids)
        .enumerate()
    {
        let Some(id) = id else {
            continue;
        };
        if field.data_type().is_nested() {
            continue;
        }
        let Some(leaf) = (0..descr.num_columns()).find(|l| descr.get_column_root_idx(*l) == root)
        else {
            continue;
        };
        let chunks: Vec<_> = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.column(leaf))
            .collect();
        let exact_bounds = !chunks.is_empty()
            && chunks.iter().all(|c| {
                c.statistics().is_some_and(|s| {
                    s.min_is_exact()
                        && s.max_is_exact()
                        && s.min_bytes_opt().is_some()
                        && s.max_bytes_opt().is_some()
                })
            });
        let bounds = if exact_bounds {
            StatisticsConverter::try_new(field.name(), arrow_schema, descr)
                .ok()
                .and_then(|converter| {
                    let reduce = |array: arrow::array::ArrayRef, keep: std::cmp::Ordering| {
                        let mut acc: Option<ScalarValue> = None;
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                return None;
                            }
                            let value = ScalarValue::try_from_array(array.as_ref(), i).ok()?;
                            acc = Some(match acc {
                                Some(current) if value.partial_cmp(&current) != Some(keep) => {
                                    current
                                },
                                _ => value,
                            });
                        }
                        acc
                    };
                    let min = reduce(
                        converter.row_group_mins(metadata.row_groups()).ok()?,
                        std::cmp::Ordering::Less,
                    )?;
                    let max = reduce(
                        converter.row_group_maxes(metadata.row_groups()).ok()?,
                        std::cmp::Ordering::Greater,
                    )?;
                    Some((min, max))
                })
        } else {
            None
        };
        out.insert(
            *id,
            ColumnFacts {
                exact_bounds: exact_bounds && bounds.is_some(),
                bounds,
                null_count: null_count(metadata, leaf),
                rows,
            },
        );
    }
    out
}

/// Refuse a partition value the file's footer does not show to be the file's:
/// see the module docs.
fn verify_partition_values(
    path: &str,
    spec: &crate::partition::PartitionSpec,
    values: &[(i32, Option<String>)],
    facts: &HashMap<i64, ColumnFacts>,
) -> Result<()> {
    use crate::partition::PartitionTransform;
    for column in &spec.columns {
        let refuse = |reason: String| -> Result<()> {
            Err(ImportError::UnverifiedPartitionValue {
                path: path.to_string(),
                column_id: column.column_id,
                reason,
            }
            .into())
        };
        let Some((_, value)) = values
            .iter()
            .find(|(index, _)| i64::from(*index) == i64::from(column.partition_key_index))
        else {
            return refuse("the file has no value for this partition key".to_string());
        };
        let Some(fact) = facts.get(&column.column_id) else {
            return refuse("the file does not hold the column".to_string());
        };
        let Some(value) = value else {
            // NULL: every value of the column NULL.
            if fact.null_count == Some(fact.rows) {
                continue;
            }
            return refuse("NULL, but the file holds values (or no null count)".to_string());
        };
        if fact.null_count != Some(0) {
            return refuse(format!(
                "{value}, but the file holds NULLs (or records no null count)"
            ));
        }
        let Some((min, max)) = &fact.bounds else {
            return refuse("the file records no exact minimum and maximum".to_string());
        };
        match &column.transform {
            PartitionTransform::Identity => {
                let parsed = ScalarValue::try_from_string(value.clone(), &min.data_type()).ok();
                if parsed.as_ref() != Some(min) || parsed.as_ref() != Some(max) {
                    return refuse(format!(
                        "{value}, but the file's values run from {min} to {max}"
                    ));
                }
            },
            PartitionTransform::Year => {
                let year_of = |v: &ScalarValue| -> Option<i32> {
                    use chrono::Datelike;
                    match v {
                        ScalarValue::Date32(Some(days)) => {
                            chrono::DateTime::from_timestamp(i64::from(*days) * 86_400, 0)
                                .map(|d| d.year())
                        },
                        ScalarValue::TimestampMicrosecond(Some(us), None) => {
                            chrono::DateTime::from_timestamp_micros(*us).map(|d| d.year())
                        },
                        _ => None,
                    }
                };
                let year: Option<i32> = value.trim().parse().ok();
                if year.is_none() || year_of(min) != year || year_of(max) != year {
                    return refuse(format!(
                        "year {value}, but the file's values run from {min} to {max}"
                    ));
                }
            },
            other => {
                return refuse(format!(
                    "a `{}` value cannot be verified from the file's footer",
                    other.to_catalog_string()
                ));
            },
        }
    }
    Ok(())
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

/// A manifest built outside the crate (`TableManifest::new`, `ManifestDataFile::new`), as
/// another table format's snapshot describes its files: one of them still carries a column
/// the table has dropped. The import adopts the file with the column ignored, and reserves
/// the dropped column's id so no column added later takes it.
#[cfg(all(test, feature = "write-sqlite"))]
mod tests {

    use std::sync::Arc;

    use crate::metadata_provider::DuckLakeTableField;
    use crate::metadata_provider_sqlite::SqliteMetadataProvider;
    use crate::metadata_writer::MetadataWriter;
    use crate::metadata_writer_sqlite::SqliteMetadataWriter;
    use crate::{
        ColumnBinding, DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, ImportError,
        ImportOptions, ManifestDataFile, TableManifest,
    };
    use arrow::array::{Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::util::display::array_value_to_string;
    use datafusion::prelude::SessionContext;
    use object_store::local::LocalFileSystem;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

    fn field(id: i64, name: &str, ty: &str) -> DuckLakeTableField {
        DuckLakeTableField {
            column_id: id,
            column_name: name.to_string(),
            column_type: ty.to_string(),
            is_nullable: true,
            parent_column: None,
        }
    }

    fn with_id(name: &str, ty: DataType, id: i64) -> Field {
        Field::new(name, ty, true).with_metadata(
            [(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect(),
        )
    }

    /// A Parquet file with columns `id` (field id 1), `name` (2) and `gone` (3).
    fn write_file(path: &std::path::Path) -> (i64, i64) {
        let schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, 1),
            with_id("name", DataType::Utf8, 2),
            with_id("gone", DataType::Utf8, 3),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        (std::fs::metadata(path).unwrap().len() as i64, 3)
    }

    async fn target(dir: &std::path::Path) -> String {
        let url = format!("sqlite:{}?mode=rwc", dir.join("target.sqlite").display());
        let data = dir.join("target");
        std::fs::create_dir_all(&data).unwrap();
        let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
        writer.set_data_path(data.to_str().unwrap()).unwrap();
        url
    }

    async fn import(url: &str, manifest: &TableManifest) -> crate::Result<()> {
        let writer = Arc::new(
            SqliteMetadataWriter::new(url)
                .await
                .unwrap()
                .with_data_inlining_row_limit(0),
        );
        let provider = Arc::new(SqliteMetadataProvider::new(url).await.unwrap());
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .import_manifest(provider, "main", "t", manifest, &ImportOptions::default())
            .await
            .map(|_| ())
    }

    async fn rows(url: &str) -> Vec<Vec<Option<String>>> {
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::new(SqliteMetadataProvider::new(url).await.unwrap()).unwrap();
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx
            .sql("SELECT * FROM lake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                out.push(
                    batch
                        .columns()
                        .iter()
                        .map(|c| {
                            c.is_valid(row)
                                .then(|| array_value_to_string(c, row).unwrap())
                        })
                        .collect(),
                );
            }
        }
        out
    }

    fn manifest(path: &std::path::Path, size: i64, count: i64) -> TableManifest {
        let file = ManifestDataFile::new(
            1,
            format!("file://{}", path.display()),
            size,
            count,
            ColumnBinding::FieldIds,
        )
        .with_deleted_positions(vec![1]);
        TableManifest::new(
            7,
            9,
            vec![field(1, "id", "int32"), field(2, "name", "varchar")],
            None,
            vec![file],
        )
    }

    /// A manifest of one file under `<dir>/<segment>/f.parquet` (columns `id`, `name`, `gone`,
    /// or `batch`'s), mapped by name, with column 4 `key` of `key_type` read from the Hive
    /// path under `key_name`.
    fn hive_manifest(
        path: &std::path::Path,
        size: i64,
        count: i64,
        key_name: &str,
        key_type: &str,
        nullable: bool,
    ) -> TableManifest {
        use crate::metadata_provider::{DuckLakeNameMapping, DuckLakeNameMappingEntry};
        let entry = |column_id: i64, source: &str, target: i64, is_partition: bool| {
            DuckLakeNameMappingEntry {
                column_id,
                source_name: source.to_string(),
                target_field_id: target,
                parent_column: None,
                is_partition,
            }
        };
        let mapping = DuckLakeNameMapping {
            mapping_id: 0,
            table_id: 0,
            mapping_type: "map_by_name".to_string(),
            entries: vec![entry(1, "id", 1, false), entry(2, key_name, 4, true)],
        };
        let file = ManifestDataFile::new(
            1,
            format!("file://{}", path.display()),
            size,
            count,
            ColumnBinding::NameMapping(mapping),
        );
        let mut key = field(4, key_name, key_type);
        key.is_nullable = nullable;
        TableManifest::new(7, 9, vec![field(1, "id", "int32"), key], None, vec![file])
    }

    /// A `NOT NULL` column a name mapping reads from the file's Hive path (#594: a Delta
    /// partition column, which no Delta file holds) imports when the path spells a value, and
    /// is refused when it spells NULL in any of the scan's spellings, or no value of its type.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_not_null_hive_partition_column_is_held_to_its_path_value() {
        for (segment, key_type, expected) in [
            ("region=eu", "varchar", Ok("eu")),
            (
                "region=__HIVE_DEFAULT_PARTITION__",
                "varchar",
                Err("reads NULL"),
            ),
            ("region=NULL", "varchar", Err("reads NULL")),
            ("region=nUlL", "varchar", Err("reads NULL")),
            ("region=", "varchar", Ok("")),
            ("n=7", "int32", Ok("7")),
            ("n=", "int32", Err("reads NULL")),
            ("n=abc", "int32", Err("has no value of its type")),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let data = dir.path().join(segment);
            std::fs::create_dir_all(&data).unwrap();
            let path = data.join("f.parquet");
            let (size, count) = write_file(&path);
            let key_name = segment.split('=').next().unwrap();
            let manifest = hive_manifest(&path, size, count, key_name, key_type, false);
            let url = target(dir.path()).await;
            let result = import(&url, &manifest).await;
            match expected {
                Ok(value) => {
                    result.unwrap_or_else(|e| panic!("{segment}: {e}"));
                    assert_eq!(
                        sql(&url, &format!("SELECT {key_name} FROM lake.main.t")).await,
                        vec![vec![Some(value.to_string())]; 3],
                        "{segment}"
                    );
                },
                Err(reason) => {
                    let e = result.expect_err(segment).to_string();
                    assert!(e.contains(reason), "{segment}: {e}");
                },
            }
        }
    }

    /// A file that also holds a column named like a Hive-path column: the scan reads the path,
    /// and the file's column binds nothing, so no statistic of it is written that would prune
    /// the path's value away.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_file_column_named_like_a_hive_column_writes_no_statistic() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("region=eu");
        std::fs::create_dir_all(&data).unwrap();
        let path = data.join("f.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("region", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["us", "us", "us"])),
            ],
        )
        .unwrap();
        let size = write_batch(&path, &batch);
        let manifest = hive_manifest(&path, size, 3, "region", "varchar", true);
        let url = target(dir.path()).await;
        import(&url, &manifest).await.unwrap();
        assert_eq!(
            catalog_rows(
                &url,
                "SELECT column_id, min_value, max_value, null_count \
                 FROM ducklake_file_column_stats WHERE column_id = 4"
            )
            .await,
            vec![],
            "no statistic of the Hive-path column"
        );
        assert_eq!(
            sql(&url, "SELECT count(*) FROM lake.main.t WHERE region = 'eu'").await,
            vec![vec![Some("3".to_string())]],
            "the path's value, not pruned by the file's"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_file_holding_a_dropped_column_imports_and_its_id_stays_taken() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;

        // Without the retired field the dropped column's id names no column: refused.
        let refused = import(&url, &manifest(&source, size, count))
            .await
            .unwrap_err();
        assert!(
            matches!(
                refused,
                DuckLakeError::Import(ImportError::FieldIdMismatch {
                    field_id: 3,
                    ..
                })
            ),
            "{refused}"
        );

        let retired =
            manifest(&source, size, count).with_retired_fields(vec![field(3, "gone", "varchar")]);
        import(&url, &retired).await.unwrap();
        assert_eq!(
            rows(&url).await,
            vec![
                vec![Some("1".into()), Some("a".into())],
                vec![Some("3".into()), Some("c".into())],
            ],
            "the dropped column is not read; the deleted position is"
        );

        // The dropped column is recorded as a column that was never visible, so its id stays
        // taken, and the id counter is past it.
        let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
        let recorded: Vec<(i64, String, i64, Option<i64>)> = sqlx::query_as(
            "SELECT column_id, column_name, begin_snapshot, end_snapshot FROM ducklake_column
             WHERE column_id = 3",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(recorded.len(), 1, "{recorded:?}");
        assert_eq!(recorded[0].1, "gone");
        assert_eq!(
            Some(recorded[0].2),
            recorded[0].3,
            "begins and ends at once"
        );
        let next: String = sqlx::query_scalar(
            "SELECT value FROM ducklake_metadata WHERE key = 'next_column_id' AND scope IS NULL",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(next.parse::<i64>().unwrap() >= 3, "next_column_id {next}");
        pool.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_retired_field_with_a_live_columns_id_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        let clash =
            manifest(&source, size, count).with_retired_fields(vec![field(2, "old", "varchar")]);
        let refused = import(&url, &clash).await.unwrap_err();
        assert!(
            matches!(
                refused,
                DuckLakeError::Import(ImportError::RetiredFieldIsLive {
                    field_id: 2,
                    ..
                })
            ),
            "{refused}"
        );
    }

    // ---- Review round 1 ---------------------------------------------------------------------

    async fn import_with(
        url: &str,
        manifest: &TableManifest,
        options: &ImportOptions,
    ) -> crate::Result<crate::ImportResult> {
        let writer = Arc::new(
            SqliteMetadataWriter::new(url)
                .await
                .unwrap()
                .with_data_inlining_row_limit(0),
        );
        let provider = Arc::new(SqliteMetadataProvider::new(url).await.unwrap());
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .import_manifest(provider, "main", "t", manifest, options)
            .await
    }

    fn write_batch(path: &std::path::Path, batch: &RecordBatch) -> i64 {
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), None)
                .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        std::fs::metadata(path).unwrap().len() as i64
    }

    fn file_at(path: &std::path::Path, size: i64, rows: i64) -> ManifestDataFile {
        ManifestDataFile::new(
            1,
            format!("file://{}", path.display()),
            size,
            rows,
            ColumnBinding::FieldIds,
        )
    }

    async fn sql(url: &str, query: &str) -> Vec<Vec<Option<String>>> {
        let ctx = SessionContext::new();
        let catalog =
            DuckLakeCatalog::new(SqliteMetadataProvider::new(url).await.unwrap()).unwrap();
        ctx.register_catalog("lake", Arc::new(catalog));
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let mut out = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                out.push(
                    batch
                        .columns()
                        .iter()
                        .map(|c| {
                            c.is_valid(row)
                                .then(|| array_value_to_string(c, row).unwrap())
                        })
                        .collect(),
                );
            }
        }
        out
    }

    async fn catalog_rows(
        url: &str,
        query: &str,
    ) -> Vec<(i64, Option<String>, Option<String>, Option<i64>)> {
        let pool = sqlx::SqlitePool::connect(url).await.unwrap();
        let rows = sqlx::query_as(sqlx::AssertSqlSafe(query.to_string()))
            .fetch_all(&pool)
            .await
            .unwrap();
        pool.close().await;
        rows
    }

    /// Finding 1 and 2: statistics and the footer length a caller supplies are not
    /// taken; the import's are the footer's.
    #[tokio::test(flavor = "multi_thread")]
    async fn supplied_statistics_and_footer_length_are_replaced_by_the_footers() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        let lying = crate::metadata_provider::DuckLakeFileColumnStatistics {
            data_file_id: 1,
            column_id: 1,
            column_size_bytes: None,
            value_count: Some(3),
            null_count: Some(0),
            min_value: Some("10".to_string()),
            max_value: Some("20".to_string()),
            contains_nan: None,
        };
        let mut manifest =
            manifest(&source, size, count).with_retired_fields(vec![field(3, "gone", "varchar")]);
        manifest.files[0] = manifest.files[0]
            .clone()
            .with_column_statistics(vec![lying])
            .with_footer_size(Some(17));
        import(&url, &manifest).await.unwrap();
        assert_eq!(
            sql(&url, "SELECT id FROM lake.main.t WHERE id = 3").await,
            vec![vec![Some("3".to_string())]],
            "a supplied bound must not prune the file"
        );
        assert_eq!(
            sql(&url, "SELECT max(id) FROM lake.main.t").await,
            vec![vec![Some("3".to_string())]]
        );
        let stats = catalog_rows(
            &url,
            "SELECT column_id, min_value, max_value, null_count FROM ducklake_file_column_stats \
             WHERE column_id = 1",
        )
        .await;
        assert_eq!(
            stats,
            vec![(1, Some("1".into()), Some("3".into()), Some(0))]
        );
        let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
        let footer: i64 = sqlx::query_scalar("SELECT footer_size FROM ducklake_data_file")
            .fetch_one(&pool)
            .await
            .unwrap();
        pool.close().await;
        assert_ne!(footer, 17, "the measured footer length is stored");
        let bytes = std::fs::read(&source).unwrap();
        let tail = &bytes[bytes.len() - 8..];
        assert_eq!(
            footer,
            i64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]))
        );
    }

    /// A bound the footer does not record exactly (parquet-rs truncates a long
    /// string's statistics, marking them inexact) is not written.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_inexact_footer_bound_is_not_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.parquet");
        let long = "z".repeat(300);
        let schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, 1),
            with_id("name", DataType::Utf8, 2),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", long.as_str()])),
            ],
        )
        .unwrap();
        let size = write_batch(&path, &batch);
        let reader = parquet::file::serialized_reader::SerializedFileReader::new(
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        let exact = parquet::file::reader::FileReader::metadata(&reader)
            .row_group(0)
            .column(1)
            .statistics()
            .unwrap()
            .max_is_exact();
        assert!(!exact, "parquet-rs truncated the long string's max");
        let url = target(dir.path()).await;
        let manifest = TableManifest::new(
            1,
            1,
            vec![field(1, "id", "int32"), field(2, "name", "varchar")],
            None,
            vec![file_at(&path, size, 2)],
        );
        import(&url, &manifest).await.unwrap();
        let stats = catalog_rows(
            &url,
            "SELECT column_id, min_value, max_value, null_count FROM ducklake_file_column_stats \
             ORDER BY column_id",
        )
        .await;
        assert_eq!(
            stats,
            vec![(1, Some("1".into()), Some("2".into()), Some(0)), (2, None, None, Some(0))]
        );
        assert_eq!(
            sql(&url, "SELECT id FROM lake.main.t WHERE name > 'zzz'").await,
            vec![vec![Some("2".into())]]
        );
    }

    /// Finding 3 and 8: after the source drops a column, re-importing with it retired
    /// succeeds; and an export of the result carries it as retired, so a round trip
    /// imports.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reimport_after_a_drop_and_its_export_carry_the_retired_column() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.parquet");
        let second = dir.path().join("second.parquet");
        let (size, count) = write_file(&first);
        write_file(&second);
        let url = target(dir.path()).await;
        let three = TableManifest::new(
            1,
            1,
            vec![field(1, "id", "int32"), field(2, "name", "varchar"), field(3, "gone", "varchar")],
            None,
            vec![file_at(&first, size, count)],
        );
        import(&url, &three).await.unwrap();
        let two = TableManifest::new(
            1,
            2,
            vec![field(1, "id", "int32"), field(2, "name", "varchar")],
            None,
            vec![file_at(&second, size, count)],
        )
        .with_retired_fields(vec![field(3, "gone", "varchar")]);
        import_with(
            &url,
            &two,
            &ImportOptions::default().with_allow_schema_evolution(true),
        )
        .await
        .unwrap();
        assert_eq!(
            sql(&url, "SELECT count(*) FROM lake.main.t").await,
            vec![vec![Some("6".into())]]
        );

        // Export: the first file still holds `gone`, which the table has dropped.
        let provider: Arc<dyn crate::MetadataProvider> =
            Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
        let head = provider.get_current_snapshot().unwrap();
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&provider), head).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        let table = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (table.as_ref() as &dyn std::any::Any)
            .downcast_ref::<crate::DuckLakeTable>()
            .unwrap();
        let exported = table.export_manifest(&ctx.state()).await.unwrap();
        assert_eq!(
            exported
                .retired_fields
                .iter()
                .map(|f| (f.column_id, f.column_name.as_str()))
                .collect::<Vec<_>>(),
            vec![(3, "gone")]
        );
        let other = tempfile::tempdir().unwrap();
        let url2 = target(other.path()).await;
        import(&url2, &exported).await.unwrap();
        assert_eq!(
            sql(&url2, "SELECT count(*) FROM lake.main.t").await,
            vec![vec![Some("6".into())]]
        );
    }

    /// Finding 4: a file column of another type than the table's, or NULLs under NOT NULL.
    #[tokio::test(flavor = "multi_thread")]
    async fn types_and_not_null_are_held_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let cases: Vec<(Vec<DuckLakeTableField>, &str)> = vec![
            // name is Utf8 in the file.
            (
                vec![field(1, "id", "int32"), field(2, "name", "int64")],
                "type",
            ),
            // id is int32 in the file: narrower is not a promotion.
            (
                vec![field(1, "id", "int16"), field(2, "name", "varchar")],
                "type",
            ),
            (
                vec![
                    field(1, "id", "int32"),
                    DuckLakeTableField {
                        is_nullable: false,
                        ..field(2, "name", "varchar")
                    },
                ],
                "ok",
            ),
            (
                vec![
                    field(1, "id", "int32"),
                    field(2, "name", "varchar"),
                    DuckLakeTableField {
                        is_nullable: false,
                        ..field(9, "later", "varchar")
                    },
                ],
                "null",
            ),
            // A wider integer DuckDB reads.
            (
                vec![field(1, "id", "int64"), field(2, "name", "varchar")],
                "ok",
            ),
        ];
        for (n, (fields, expect)) in cases.into_iter().enumerate() {
            let sub = dir.path().join(format!("c{n}"));
            std::fs::create_dir_all(&sub).unwrap();
            let url = target(&sub).await;
            let manifest =
                TableManifest::new(1, 1, fields, None, vec![file_at(&source, size, count)])
                    .with_retired_fields(vec![field(3, "gone", "varchar")]);
            let result = import(&url, &manifest).await;
            match expect {
                "ok" => result.unwrap(),
                "type" => assert!(
                    matches!(
                        result,
                        Err(DuckLakeError::Import(
                            ImportError::ColumnTypeMismatch { .. }
                        ))
                    ),
                    "{n}: {result:?}"
                ),
                _ => assert!(
                    matches!(
                        result,
                        Err(DuckLakeError::Import(ImportError::NotNullViolated { .. }))
                    ),
                    "{n}: {result:?}"
                ),
            }
        }

        // NULLs in a NOT NULL column.
        let nulls = dir.path().join("nulls.parquet");
        let schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, 1),
            with_id("name", DataType::Utf8, 2),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![Some("a"), None])),
            ],
        )
        .unwrap();
        let size = write_batch(&nulls, &batch);
        let sub = dir.path().join("nulls");
        std::fs::create_dir_all(&sub).unwrap();
        let url = target(&sub).await;
        let manifest = TableManifest::new(
            1,
            1,
            vec![
                field(1, "id", "int32"),
                DuckLakeTableField {
                    is_nullable: false,
                    ..field(2, "name", "varchar")
                },
            ],
            None,
            vec![file_at(&nulls, size, 2)],
        );
        let result = import(&url, &manifest).await;
        assert!(
            matches!(
                result,
                Err(DuckLakeError::Import(ImportError::NotNullViolated { .. }))
            ),
            "{result:?}"
        );
    }

    /// Finding 5: retired fields that cannot be ones.
    #[tokio::test(flavor = "multi_thread")]
    async fn invalid_retired_fields_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        let gone = field(3, "gone", "varchar");
        for retired in [
            vec![gone.clone(), gone.clone()],
            vec![gone.clone(), field(2_147_483_540, "row_id", "int64")],
            vec![gone.clone(), field(0, "zero", "int32")],
            vec![
                gone.clone(),
                DuckLakeTableField {
                    parent_column: Some(77),
                    ..field(5, "orphan", "int32")
                },
            ],
        ] {
            let manifest = manifest(&source, size, count).with_retired_fields(retired.clone());
            let result = import(&url, &manifest).await;
            assert!(
                matches!(
                    result,
                    Err(DuckLakeError::Import(
                        ImportError::InvalidRetiredField { .. }
                    ))
                ),
                "{retired:?}: {result:?}"
            );
        }
    }

    /// Finding 6: a dropped struct member's id is refused unless retired.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_dropped_struct_member_must_be_retired() {
        use arrow::array::StructArray;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested.parquet");
        let x = with_id("x", DataType::Int32, 4);
        let y = with_id("y", DataType::Int32, 5);
        let s = StructArray::new(
            vec![Arc::new(x), Arc::new(y)].into(),
            vec![Arc::new(Int32Array::from(vec![1])), Arc::new(Int32Array::from(vec![2]))],
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, 1),
            with_id("s", arrow::array::Array::data_type(&s).clone(), 3),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![7])), Arc::new(s)],
        )
        .unwrap();
        let size = write_batch(&path, &batch);
        let fields = vec![
            field(1, "id", "int32"),
            field(3, "s", "struct"),
            DuckLakeTableField {
                parent_column: Some(3),
                ..field(4, "x", "int32")
            },
        ];
        let url = target(dir.path()).await;
        let without = TableManifest::new(1, 1, fields.clone(), None, vec![file_at(&path, size, 1)]);
        let refused = import(&url, &without).await.unwrap_err();
        assert!(
            matches!(
                refused,
                DuckLakeError::Import(ImportError::FieldIdMismatch {
                    field_id: 5,
                    ..
                })
            ),
            "{refused}"
        );
        let with = without.with_retired_fields(vec![DuckLakeTableField {
            parent_column: Some(3),
            ..field(5, "y", "int32")
        }]);
        import(&url, &with).await.unwrap();
        assert_eq!(
            sql(&url, "SELECT id, s.x FROM lake.main.t").await,
            vec![vec![Some("7".into()), Some("1".into())]]
        );
    }

    /// Finding 7: a partition value kept under a matching spec is verified against
    /// the file; a wrong one is refused, the right one kept.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_partition_value_the_file_does_not_hold_is_refused() {
        use crate::partition::{PartitionSpec, PartitionSpecColumn, PartitionTransform};
        let dir = tempfile::tempdir().unwrap();
        let one = |name: &str, part: i32| {
            let path = dir.path().join(name);
            let schema = Arc::new(Schema::new(vec![
                with_id("id", DataType::Int32, 1),
                with_id("part", DataType::Int32, 2),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(vec![1, 2])),
                    Arc::new(Int32Array::from(vec![part, part])),
                ],
            )
            .unwrap();
            let size = write_batch(&path, &batch);
            (path, size)
        };
        let fields = vec![field(1, "id", "int32"), field(2, "part", "int32")];
        let url = target(dir.path()).await;
        let (first, size) = one("first.parquet", 1);
        let created = import_with(
            &url,
            &TableManifest::new(1, 1, fields.clone(), None, vec![file_at(&first, size, 2)]),
            &ImportOptions::default(),
        )
        .await
        .unwrap();
        SqliteMetadataWriter::new(&url)
            .await
            .unwrap()
            .set_partition_spec(
                created.table_id,
                &[("part".to_string(), PartitionTransform::Identity)],
            )
            .unwrap();
        let spec = PartitionSpec {
            partition_id: 50,
            columns: vec![PartitionSpecColumn {
                partition_key_index: 0,
                column_id: 2,
                transform: PartitionTransform::Identity,
            }],
            prune_safe: true,
        };
        let (second, size) = one("second.parquet", 4);
        for (value, ok) in [("5", false), ("4", true)] {
            let manifest = TableManifest::new(
                1,
                2,
                fields.clone(),
                Some(spec.clone()),
                vec![
                    file_at(&second, size, 2)
                        .with_partition(Some(50), vec![(0, Some(value.to_string()))]),
                ],
            );
            let result = import(&url, &manifest).await;
            if ok {
                result.unwrap();
            } else {
                assert!(
                    matches!(
                        result,
                        Err(DuckLakeError::Import(
                            ImportError::UnverifiedPartitionValue { .. }
                        ))
                    ),
                    "{result:?}"
                );
            }
        }
        assert_eq!(
            sql(&url, "SELECT count(*) FROM lake.main.t WHERE part = 4").await,
            vec![vec![Some("2".into())]]
        );
        let kept = catalog_rows(
            &url,
            "SELECT data_file_id, partition_value, partition_value, NULL FROM ducklake_file_partition_value",
        )
        .await;
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].1.as_deref(), Some("4"));
    }

    // ---- Review round 2 ---------------------------------------------------------------------

    /// Finding 1: a float32 file under a float64 column keeps no bounds (the
    /// float32 `0.1` would prune the float64 `0.10000000149011612` away); an int32
    /// file under an int64 column keeps its exact ones.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_promoted_float_file_keeps_no_bounds_and_its_values_are_found() {
        use arrow::array::Float32Array;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("float.parquet");
        let schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, 1),
            with_id("f", DataType::Float32, 2),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Float32Array::from(vec![0.1, 0.5])),
            ],
        )
        .unwrap();
        let size = write_batch(&path, &batch);
        let url = target(dir.path()).await;
        let manifest = TableManifest::new(
            1,
            1,
            vec![field(1, "id", "int64"), field(2, "f", "float64")],
            None,
            vec![file_at(&path, size, 2)],
        );
        import(&url, &manifest).await.unwrap();
        let stats = catalog_rows(
            &url,
            "SELECT column_id, min_value, max_value, null_count FROM ducklake_file_column_stats \
             ORDER BY column_id",
        )
        .await;
        assert_eq!(
            stats,
            vec![(1, Some("1".into()), Some("2".into()), Some(0)), (2, None, None, Some(0))],
            "the widened int keeps its bounds, the widened float none"
        );
        for (filter, expected) in [
            ("f = CAST(CAST(0.1 AS FLOAT) AS DOUBLE)", vec!["1"]),
            ("f = 0.10000000149011612", vec!["1"]),
            ("f > 0.1", vec!["1", "2"]),
            ("f < 0.2", vec!["1"]),
            ("id = 2", vec!["2"]),
        ] {
            let got: Vec<String> = sql(
                &url,
                &format!("SELECT id FROM lake.main.t WHERE {filter} ORDER BY id"),
            )
            .await
            .into_iter()
            .map(|row| row[0].clone().unwrap())
            .collect();
            assert_eq!(got, expected, "{filter}");
        }
    }

    /// Finding 2: an id an import reserved (a never-visible row) is carried by an
    /// export as retired, so the export imports again.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_export_carries_an_imported_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        import(
            &url,
            &manifest(&source, size, count).with_retired_fields(vec![field(3, "gone", "varchar")]),
        )
        .await
        .unwrap();
        let provider: Arc<dyn crate::MetadataProvider> =
            Arc::new(SqliteMetadataProvider::new(&url).await.unwrap());
        let head = provider.get_current_snapshot().unwrap();
        let catalog = DuckLakeCatalog::with_snapshot(Arc::clone(&provider), head).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        let table = ctx
            .catalog("lake")
            .unwrap()
            .schema("main")
            .unwrap()
            .table("t")
            .await
            .unwrap()
            .unwrap();
        let table = (table.as_ref() as &dyn std::any::Any)
            .downcast_ref::<crate::DuckLakeTable>()
            .unwrap();
        let exported = table.export_manifest(&ctx.state()).await.unwrap();
        assert_eq!(
            exported
                .retired_fields
                .iter()
                .map(|f| (f.column_id, f.column_name.as_str()))
                .collect::<Vec<_>>(),
            vec![(3, "gone")]
        );
        let other = tempfile::tempdir().unwrap();
        let url2 = target(other.path()).await;
        import(&url2, &exported).await.unwrap();
        assert_eq!(rows(&url2).await.len(), 2);
    }

    /// Finding 4: a nested table column over a file column of another kind.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_nested_column_over_a_scalar_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        let child = |id: i64, name: &str| DuckLakeTableField {
            parent_column: Some(2),
            ..field(id, name, "int32")
        };
        for (kind, children) in [
            ("struct", vec![child(10, "x")]),
            ("list", vec![child(10, "element")]),
            ("map", vec![child(10, "key"), child(11, "value")]),
        ] {
            let mut fields = vec![field(1, "id", "int32"), field(2, "name", kind)];
            fields.extend(children);
            let manifest =
                TableManifest::new(1, 1, fields, None, vec![file_at(&source, size, count)])
                    .with_retired_fields(vec![field(3, "gone", "varchar")]);
            let refused = import(&url, &manifest).await.unwrap_err();
            assert!(
                matches!(
                    refused,
                    DuckLakeError::Import(ImportError::ColumnTypeMismatch { .. })
                ),
                "{kind}: {refused}"
            );
        }
    }

    /// Finding 3: `validate_manifest` refuses what the import refuses (it is the
    /// import's own checks), accepts what the import accepts, and writes nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn validation_refuses_as_the_import_does_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.parquet");
        let (size, count) = write_file(&source);
        let url = target(dir.path()).await;
        let db = dir.path().join("target.sqlite");
        let provider = SqliteMetadataProvider::new(&url).await.unwrap();
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
        let before = std::fs::read(&db).unwrap();
        let refused = crate::validate_manifest(
            &store,
            &provider,
            "main",
            "t",
            &manifest(&source, size, count),
            &ImportOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(
                refused,
                DuckLakeError::Import(ImportError::FieldIdMismatch {
                    field_id: 3,
                    ..
                })
            ),
            "{refused}"
        );
        let good =
            manifest(&source, size, count).with_retired_fields(vec![field(3, "gone", "varchar")]);
        crate::validate_manifest(
            &store,
            &provider,
            "main",
            "t",
            &good,
            &ImportOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read(&db).unwrap(),
            before,
            "validation writes nothing"
        );

        // And the import, after it, accepts it.
        import(&url, &good).await.unwrap();
        assert_eq!(rows(&url).await.len(), 2);
    }
}
