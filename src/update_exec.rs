//! DuckLake `UPDATE` execution plan.
//!
//! [`DuckLakeUpdateExec`] is the physical operator DataFusion's planner lowers
//! `UPDATE t SET col = expr ... [WHERE p]` onto (via
//! [`TableProvider::update`](datafusion::catalog::TableProvider::update)). All
//! work is deferred to execute time so planning / `EXPLAIN` never mutate data:
//!
//! 1. For each source data file that may hold matching rows, collect its
//!    pre-built positional scan, select the rows matching the predicate, apply
//!    the assignments, and produce rewritten row versions that RETAIN each row's
//!    original rowid (written into NEW data file(s) that embed the rowid column,
//!    so lineage survives the rewrite). On a partitioned table each rewritten row
//!    is routed by its OWN post-assignment key values, so an assignment that
//!    changes a partition key MOVES the row to its new partition; the rewrite then
//!    spans one file per output partition.
//! 2. Delete the superseded row versions from each source file.
//! 3. Commit ATOMICALLY: the new row versions and every deletion land in ONE
//!    snapshot.
//! 4. Yield a single row `count: UInt64` = rows updated.
//!
//! On a writer that [stages row mutations](MetadataWriter::supports_inlined_row_mutations)
//! the statement follows official DuckLake's UPDATE, a delete of the old row
//! versions plus an insert of the new ones that keep their row ids
//! (`ducklake_update.cpp`): visible inlined rows are updated too, their old
//! versions ended; each source file's superseded rows are deleted as a `DELETE`
//! deletes them, so at most the scoped `data_inlining_row_limit` are recorded in
//! `ducklake_inlined_delete_<table_id>`, a file the statement empties is
//! dropped, and a live delete file is replaced by one carrying each position's
//! snapshot; and when the statement writes at most the limit's rows, the new
//! versions are inlined with their row ids rather than written to Parquet. A
//! statement needing any of these commits as one staged write
//! ([`MetadataWriter::commit_multi_table`]); otherwise the rewritten files and
//! new delete files commit through
//! [`MetadataWriter::register_data_files_with_deletes`], driven by
//! `TableWriteSession::finish_with_deletes`. Any other writer gets each source
//! file's cumulative delete file (superseded rows unioned with the already
//! deleted ones).
//!
//! The new versions stream: each source file is read batch by batch, and its
//! rewritten rows are held only while the statement's count of them is at most
//! the inlining limit (whether to inline them is decided on the whole
//! statement); past it they go into the write session as they are made, its
//! Parquet out in parts as it is encoded (ADR-0037 in ferrion). What is still
//! held whole is each file's superseded positions, which its delete is made
//! of. Runs in a single DataFusion output partition.
//!
//! # Session lifecycle
//!
//! A catalog from [`DuckLakeCatalog::with_writer`](crate::DuckLakeCatalog::with_writer) reads
//! the latest snapshot for each statement, so a later statement in the same `SessionContext`
//! sees this `UPDATE` and any other writer's commit. A statement plans against one snapshot; a
//! commit by another writer between planning and execution aborts with a
//! [`Conflict`](crate::DuckLakeError::Conflict) and leaves the catalog unchanged.

use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::apply_expression_roots;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, TryStreamExt};

use crate::compaction::sorted_rewrite_batches;
use crate::delete_exec::{DeletionTarget, FileDeletions};
use crate::metadata_writer::{
    DataFileInfo, DeleteFileEntry, InlinedRowRef, MetadataWriter, StagedTableData, WriteMode,
};
use crate::table::{DuckLakeTable, UpdateSourceScan};
use crate::table_writer::TableWriteSession;
use crate::table_writer::{DuckLakeTableWriter, validate_not_null_batches};

/// Schema for the output of update operations (count of rows updated).
fn make_update_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that applies an `UPDATE` to a DuckLake table.
pub struct DuckLakeUpdateExec {
    /// Read-only handle to the target table, used to turn each source file's
    /// collected scan batches into rewritten rows at execute time.
    table: Arc<DuckLakeTable>,
    /// Metadata writer for the atomic append-with-deletes commit.
    writer: Arc<dyn MetadataWriter>,
    schema_name: String,
    table_name: String,
    base_snapshot: i64,
    /// Per-source-file positional read plans (built at plan time).
    scans: Vec<UpdateSourceScan>,
    /// `(physical_column_index, new_value_expr)` for each assigned column.
    assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    /// AND of the WHERE predicates, or `None` to update all rows.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    object_store_url: Arc<ObjectStoreUrl>,
    cache: Arc<PlanProperties>,
}

impl DuckLakeUpdateExec {
    /// Create a new `DuckLakeUpdateExec`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: Arc<DuckLakeTable>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        base_snapshot: i64,
        scans: Vec<UpdateSourceScan>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        object_store_url: Arc<ObjectStoreUrl>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            table,
            writer,
            schema_name,
            table_name,
            base_snapshot,
            scans,
            assignments,
            predicate,
            object_store_url,
            cache,
        }
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_update_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeUpdateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeUpdateExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("source_files", &self.scans.len())
            .field("assignments", &self.assignments.len())
            .field("has_predicate", &self.predicate.is_some())
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeUpdateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeUpdateExec: schema={}, table={}, assignments={}, where={}",
                    self.schema_name,
                    self.table_name,
                    self.assignments.len(),
                    if self.predicate.is_some() {
                        "yes"
                    } else {
                        "no"
                    }
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeUpdateExec {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        let assignments = self.assignments.iter().map(|(_, expr)| expr);
        apply_expression_roots(assignments.chain(self.predicate.iter()), f)
    }

    fn name(&self) -> &str {
        "DuckLakeUpdateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    /// No DataFusion children: the per-file source scans are internal and are
    /// executed directly at execute time, so the optimizer treats this as a
    /// leaf and never rewrites (e.g. repartitions) those scans.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "DuckLakeUpdateExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DuckLakeUpdateExec only supports partition 0, got {partition}"
            )));
        }

        let table = Arc::clone(&self.table);
        let writer = Arc::clone(&self.writer);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let base_snapshot = self.base_snapshot;
        let scans = self.scans.clone();
        let assignments = self.assignments.clone();
        let predicate = self.predicate.clone();
        let object_store_url = self.object_store_url.clone();
        let output_schema = make_update_count_schema();

        let stream = stream::once(async move {
            let object_store = context
                .runtime_env()
                .object_store(object_store_url.as_ref())?;
            let inline_capable = writer.supports_inlined_row_mutations();
            let table_writer = DuckLakeTableWriter::new(writer, object_store)
                .map(|writer| writer.with_options(table.write_options()))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Official DuckLake plans an UPDATE as a delete of the old row versions
            // plus an insert of the new ones that keeps their row ids
            // (`ducklake_update.cpp`), both following the scoped
            // `data_inlining_row_limit`: a source file's superseded rows become
            // inlined deletions when they number at most the limit, and the new
            // versions are inlined when the whole statement writes at most the
            // limit.
            let inline_limit = if inline_capable {
                table_writer
                    .table_data_inlining_row_limit(&schema_name, &table_name)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?
            } else {
                0
            };

            // Rewrite each source file's matching rows as they are read, handing
            // the new versions to `versions`: held while they number at most the
            // limit, and past it written as they come (see the module docs). The
            // superseded rows' deletes are made once every file is read; the
            // catalog commit that makes everything visible happens once,
            // atomically, below, so a failure before it leaves the live snapshot
            // untouched, and the write session is aborted, removing what it
            // uploaded.
            let mut versions = NewVersionSink::new(
                &table_writer,
                &schema_name,
                &table_name,
                table.physical_schema(),
                base_snapshot,
                inline_limit,
                context.memory_pool(),
            );
            let mut pending_deletes = Vec::new();
            let mut superseded_positions: Vec<(&UpdateSourceScan, Vec<i64>)> = Vec::new();
            let mut superseded_inlined: Vec<InlinedRowRef> = Vec::new();
            let mut total_updated: u64 = 0;
            // The superseded positions are held until the commit, which makes the
            // deletes of them: charged to the pool as they are taken, so an update
            // too large to hold them is refused rather than outgrowing it.
            let positions_held =
                datafusion::execution::memory_pool::MemoryConsumer::new("DuckLakeUpdateExec")
                    .register(context.memory_pool());

            // Boxed, as is the commit below: a session is kilobytes, and this
            // future nests in every caller's.
            let read = Box::pin(async {
                let out_schema = table.update_output_schema();
                for scan in &scans {
                    let mut batches = datafusion::physical_plan::execute_stream(
                        Arc::clone(&scan.scan),
                        context.clone(),
                    )?;
                    let mut positions = Vec::new();
                    while let Some(batch) = batches.try_next().await? {
                        let Some(rewritten) = table.apply_update_to_batch(
                            scan,
                            &out_schema,
                            &batch,
                            predicate.as_ref(),
                            &assignments,
                        )?
                        else {
                            continue;
                        };
                        positions_held
                            .try_grow(rewritten.positions.len() * std::mem::size_of::<i64>())?;
                        positions.extend(rewritten.positions);
                        versions.push(rewritten.batch).await?;
                    }
                    if positions.is_empty() {
                        continue;
                    }
                    total_updated += positions.len() as u64;
                    if inline_capable {
                        positions.sort_unstable();
                        superseded_positions.push((scan, positions));
                        continue;
                    }
                    let cumulative = DuckLakeTable::cumulative_positions(scan, positions);
                    pending_deletes.push((
                        scan.data_file_id,
                        scan.delete_file_id,
                        scan.source_path.clone(),
                        cumulative,
                    ));
                }

                if inline_capable {
                    let inlined = table
                        .inlined_data_with_row_ids()
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    for rows in &inlined {
                        if let Some((batch, superseded)) =
                            table.apply_update_to_inlined(rows, predicate.as_ref(), &assignments)?
                        {
                            total_updated += batch.num_rows() as u64;
                            versions.push(batch).await?;
                            superseded_inlined.extend(superseded);
                        }
                    }
                }
                Ok::<(), DataFusionError>(())
            })
            .await;
            if let Err(e) = read {
                versions.abort().await;
                return Err(e);
            }

            // No matching rows: genuine no-op, publish nothing (and no session
            // was begun: one begins only past a row).
            if total_updated == 0 {
                let count: ArrayRef = Arc::new(UInt64Array::from(vec![0u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count])?);
            }

            commit_row_versions(
                &RowVersionCommit {
                    table: &table,
                    table_writer: &table_writer,
                    context: Arc::clone(&context),
                    schema_name: &schema_name,
                    table_name: &table_name,
                    base_snapshot,
                    inline_capable,
                    inline_limit,
                },
                versions.finish(),
                superseded_positions,
                pending_deletes,
                superseded_inlined,
                total_updated,
            )
            .await?;
            // Held until the deletes made of them are committed.
            drop(positions_held);
            let count: ArrayRef = Arc::new(UInt64Array::from(vec![total_updated]));
            Ok(RecordBatch::try_new(output_schema, vec![count])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_update_count_schema(),
            stream.map_err(|e: DataFusionError| e),
        )))
    }
}

/// What [`commit_row_versions`] commits for: the target table as read, the
/// writer the commit goes through, and the statement's inlining limit.
pub(crate) struct RowVersionCommit<'a> {
    pub(crate) table: &'a DuckLakeTable,
    pub(crate) table_writer: &'a DuckLakeTableWriter,
    pub(crate) context: Arc<TaskContext>,
    pub(crate) schema_name: &'a str,
    pub(crate) table_name: &'a str,
    pub(crate) base_snapshot: i64,
    /// Whether the writer stages row mutations
    /// ([`MetadataWriter::supports_inlined_row_mutations`]).
    pub(crate) inline_capable: bool,
    /// The statement's scoped `data_inlining_row_limit` (`0` on a writer that
    /// does not stage row mutations).
    pub(crate) inline_limit: usize,
}

/// An `UPDATE`'s new row versions, `[physical columns..., rowid]`, each keeping
/// its row's id: held, or already written into a session
/// ([`NewVersionSink`]).
pub(crate) enum NewVersions {
    /// Every new version, in memory: at most the inlining limit's rows from a
    /// streaming `UPDATE`, or a merge's (`keyed_mutation`), which holds them.
    Held(Vec<RecordBatch>),
    /// More than the limit's rows, written into this session as they were made
    /// and not yet finished.
    Written(Box<TableWriteSession>),
}

/// Holds an `UPDATE`'s new versions while they number at most the inlining
/// limit, and past it begins the session that writes them and writes each as
/// it is made: the held ones first, so the rows keep the order they were read
/// in (official's `UPDATE` plans no sort, whatever `sort_on_insert` says).
struct NewVersionSink<'a> {
    table_writer: &'a DuckLakeTableWriter,
    schema_name: &'a str,
    table_name: &'a str,
    physical_schema: SchemaRef,
    base_snapshot: i64,
    limit: usize,
    held: Vec<RecordBatch>,
    rows: usize,
    /// Boxed: a session is kilobytes, and this sink lives in the UPDATE's future.
    session: Option<Box<TableWriteSession>>,
    /// What the held versions or the session hold in memory, charged to the
    /// pool the statement's operators use, as an insert's is
    /// ([`crate::insert_exec`]).
    charged: datafusion::execution::memory_pool::MemoryReservation,
}

impl<'a> NewVersionSink<'a> {
    fn new(
        table_writer: &'a DuckLakeTableWriter,
        schema_name: &'a str,
        table_name: &'a str,
        physical_schema: SchemaRef,
        base_snapshot: i64,
        limit: usize,
        pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
    ) -> Self {
        Self {
            charged: datafusion::execution::memory_pool::MemoryConsumer::new("DuckLakeUpdateExec")
                .register(pool),
            table_writer,
            schema_name,
            table_name,
            physical_schema,
            base_snapshot,
            limit,
            held: Vec::new(),
            rows: 0,
            session: None,
        }
    }

    /// Take one batch of new versions. A NULL in a NOT NULL column is refused
    /// here, before the batch is written and before any delete is made; so is
    /// a batch past the pool.
    async fn push(&mut self, batch: RecordBatch) -> DataFusionResult<()> {
        self.write(batch).await?;
        let held = match &self.session {
            Some(session) => session.memory_size(),
            None => self
                .held
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum(),
        };
        self.charged.try_resize(held)
    }

    async fn write(&mut self, batch: RecordBatch) -> DataFusionResult<()> {
        validate_not_null_batches(self.physical_schema.as_ref(), std::slice::from_ref(&batch))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        self.rows += batch.num_rows();
        let session = match &mut self.session {
            Some(session) => session,
            None if self.rows <= self.limit => {
                self.held.push(batch);
                return Ok(());
            },
            None => {
                let session = self
                    .table_writer
                    .begin_write_with_embedded_rowid(
                        self.schema_name,
                        self.table_name,
                        self.physical_schema.as_ref(),
                        WriteMode::Append,
                    )
                    .map(|session| Box::new(session.with_base_snapshot_id(self.base_snapshot)))
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let session = self.session.insert(session);
                for held in std::mem::take(&mut self.held) {
                    Box::pin(session.write_batch_async(&held))
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                session
            },
        };
        // Boxed, as an insert's are (`insert_exec::stream_into`).
        Box::pin(session.write_batch_async(&batch))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Every new version taken: the held ones, or the session they went into.
    fn finish(self) -> NewVersions {
        match self.session {
            Some(session) => NewVersions::Written(session),
            None => NewVersions::Held(self.held),
        }
    }

    /// Abandon the new versions: a session begun is aborted, which removes what
    /// it uploaded and releases what its start inserted.
    async fn abort(self) {
        if let Some(session) = self.session {
            abort_session(*session).await;
        }
    }
}

/// Abort `session`, logging rather than returning a failure to: the caller is
/// already returning the error that ended the write.
async fn abort_session(session: TableWriteSession) {
    if let Err(error) = session.abort().await {
        tracing::warn!(error = %error, "failed to abort an UPDATE's write session");
    }
}

/// Commit an `UPDATE`'s new row versions and the deletion of the versions they
/// supersede in one snapshot, as official's `UPDATE` does (see the module
/// docs): `versions` are the new versions ([`NewVersions`]); a session they
/// were written into is finished here, or aborted when the commit fails before
/// it is; `superseded_positions` are each source file's superseded rows
/// on a writer that stages row mutations, `pending_deletes` each source
/// file's cumulative delete file `(data_file_id, live delete file, path,
/// positions)` on any other, and `superseded_inlined` the inlined rows the new
/// versions replace. Returns the committed snapshot.
pub(crate) async fn commit_row_versions(
    commit: &RowVersionCommit<'_>,
    versions: NewVersions,
    superseded_positions: Vec<(&UpdateSourceScan, Vec<i64>)>,
    pending_deletes: Vec<(i64, Option<i64>, String, Vec<i64>)>,
    superseded_inlined: Vec<InlinedRowRef>,
    total_updated: u64,
) -> DataFusionResult<Option<i64>> {
    let (updated_batches, mut session) = match versions {
        NewVersions::Held(batches) => (batches, None),
        NewVersions::Written(session) => (Vec::new(), Some(*session)),
    };
    let committed = Box::pin(commit_new_versions(
        commit,
        updated_batches,
        &mut session,
        superseded_positions,
        pending_deletes,
        superseded_inlined,
        total_updated,
    ))
    .await;
    // Still here only when the commit failed before it took the session over.
    if let Some(session) = session {
        abort_session(session).await;
    }
    committed
}

/// [`commit_row_versions`] over the held versions `updated_batches` (none when
/// they were written into `session`, which this takes once it finishes it).
async fn commit_new_versions(
    commit: &RowVersionCommit<'_>,
    updated_batches: Vec<RecordBatch>,
    session: &mut Option<TableWriteSession>,
    superseded_positions: Vec<(&UpdateSourceScan, Vec<i64>)>,
    pending_deletes: Vec<(i64, Option<i64>, String, Vec<i64>)>,
    superseded_inlined: Vec<InlinedRowRef>,
    total_updated: u64,
) -> DataFusionResult<Option<i64>> {
    let RowVersionCommit {
        table,
        table_writer,
        context,
        schema_name,
        table_name,
        base_snapshot,
        inline_capable,
        inline_limit,
    } = commit;
    let (table, table_writer, schema_name, table_name) =
        (*table, *table_writer, *schema_name, *table_name);
    let (base_snapshot, inline_capable, inline_limit) =
        (*base_snapshot, *inline_capable, *inline_limit);
    let context = Arc::clone(context);
    let physical_schema = table.physical_schema();
    validate_not_null_batches(physical_schema.as_ref(), &updated_batches)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Each source file's superseded rows are deleted as official's
    // `DuckLakeDelete::FlushDelete` deletes them.
    let mut deletions = FileDeletions::default();
    for (scan, positions) in superseded_positions {
        let target = DeletionTarget {
            data_file_id: scan.data_file_id,
            data_file_path: &scan.source_path,
            record_count: scan.record_count,
            delete_file: scan
                .delete_file_id
                .map(|id| (id, &scan.existing_parquet_deletes)),
        };
        deletions
            .record(
                table,
                table_writer,
                schema_name,
                table_name,
                &target,
                positions,
                inline_limit,
                base_snapshot,
            )
            .await?;
    }
    let mut delete_entries: Vec<DeleteFileEntry> = std::mem::take(&mut deletions.positional);
    for (data_file_id, expected_prev_delete_file, source_path, positions) in pending_deletes {
        let delete_info = table_writer
            .write_delete_file(schema_name, table_name, &source_path, &positions)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        delete_entries.push(DeleteFileEntry {
            data_file_id,
            expected_prev_delete_file,
            delete: delete_info,
        });
    }

    // The new versions without their rowid column, in the table's types.
    let new_versions = updated_batches
        .iter()
        .map(|batch| {
            let columns = batch.columns()[..batch.num_columns() - 1].to_vec();
            RecordBatch::try_new(Arc::clone(&physical_schema), columns)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // Written versions are past the limit, so never inlined.
    let inline_new_versions = session.is_none()
        && total_updated <= inline_limit as u64
        && table_writer
            .should_inline(
                schema_name,
                table_name,
                total_updated as usize,
                physical_schema.as_ref(),
                &new_versions,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let staged = inline_capable
        && (inline_new_versions
            || deletions.needs_staged_commit()
            || !superseded_inlined.is_empty());

    // Append the rewritten rows (embedding their original rowids) AND
    // apply every positional delete in ONE snapshot. The new versions keep
    // the order they were read in: official's UPDATE plans no sort
    // (`DuckLakeCatalog::PlanUpdate`), whatever `sort_on_insert` says.

    if staged {
        let (data, row_ids) = if inline_new_versions {
            let row_ids = updated_batches
                .iter()
                .flat_map(|batch| {
                    let ids = batch
                        .column(batch.num_columns() - 1)
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .expect("rowid column is Int64")
                        .clone();
                    ids.values().to_vec()
                })
                .collect::<Vec<_>>();
            (StagedTableData::Inlined(new_versions), Some(row_ids))
        } else {
            let files = match session.take() {
                Some(session) => session
                    .finish_files()
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
                None => {
                    write_rewritten_files(
                        table_writer,
                        Arc::clone(&context),
                        schema_name,
                        table_name,
                        physical_schema.as_ref(),
                        base_snapshot,
                        updated_batches,
                    )
                    .await?
                },
            };
            (StagedTableData::Files(files), None)
        };
        let state = datafusion::execution::SessionStateBuilder::new()
            .with_config(context.session_config().clone())
            .with_runtime_env(context.runtime_env())
            .build();
        let latest = crate::delete_exec::LatestDeleteFiles {
            table,
            state: &state,
        };
        let mut transaction = table_writer.transaction();
        transaction.set_delete_file_positions(&latest);
        transaction
            .stage_row_mutation(
                schema_name,
                table_name,
                table.table_id(),
                table.column_ids(),
                physical_schema.as_ref(),
                base_snapshot,
                data,
                row_ids,
                delete_entries,
                deletions.rewritten,
                superseded_inlined,
                deletions.inlined,
                deletions.dropped,
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let committed = transaction
            .commit()
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        return Ok(committed.first().map(|result| result.snapshot_id));
    }

    let session = match session.take() {
        Some(session) => session,
        None => {
            let mut updated_batches =
                sorted_rewrite_batches(Arc::clone(&context), updated_batches, None)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let mut session = table_writer
                .begin_write_with_embedded_rowid(
                    schema_name,
                    table_name,
                    physical_schema.as_ref(),
                    WriteMode::Append,
                )
                .map(|session| session.with_base_snapshot_id(base_snapshot))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            while let Some(batch) = updated_batches.try_next().await? {
                if let Err(e) = session.write_batch_async(&batch).await {
                    abort_session(session).await;
                    return Err(DataFusionError::External(Box::new(e)));
                }
            }
            session
        },
    };
    // The session removes the delete files with its own when the commit
    // definitely registered nothing.
    let committed = session
        .finish_with_deletes(&delete_entries)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(Some(committed.snapshot_id))
}

/// Write the rewritten rows `[physical columns..., rowid]` to Parquet files
/// that embed each row's row id, in the table's partition layout and the
/// order they arrive, without committing them.
#[allow(clippy::too_many_arguments)]
async fn write_rewritten_files(
    table_writer: &DuckLakeTableWriter,
    context: Arc<TaskContext>,
    schema_name: &str,
    table_name: &str,
    physical_schema: &Schema,
    base_snapshot: i64,
    batches: Vec<RecordBatch>,
) -> DataFusionResult<Vec<DataFileInfo>> {
    let mut batches = sorted_rewrite_batches(context, batches, None)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let mut session = table_writer
        .begin_write_with_embedded_rowid(
            schema_name,
            table_name,
            physical_schema,
            WriteMode::Append,
        )
        .map(|session| session.with_base_snapshot_id(base_snapshot))
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    while let Some(batch) = batches.try_next().await? {
        session
            .write_batch_async(&batch)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }
    session
        .finish_files()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_count_schema() {
        let schema = make_update_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }

    /// A keyed `UPDATE` hands its predicate to each source file's reader, so a
    /// multi-row-group file decodes only the row groups that can match.
    #[cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]
    mod pruning {
        use std::sync::Arc;

        use arrow::array::{Float64Array, Int32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion::physical_plan::{collect, displayable};
        use datafusion::prelude::SessionContext;
        use object_store::local::LocalFileSystem;

        use super::super::DuckLakeUpdateExec;
        use crate::metadata_writer::MetadataWriter;
        use crate::{
            DuckLakeCatalog, DuckLakeTableWriter, SqliteMetadataProvider, SqliteMetadataWriter,
        };

        const ROWS: i32 = 40;
        const ROW_GROUP: usize = 4;

        /// One data file of `ROWS` rows `(id, score)` in `ROW_GROUP`-row groups.
        async fn seed(dir: &tempfile::TempDir) -> String {
            let conn = format!("sqlite:{}?mode=rwc", dir.path().join("t.db").display());
            let data = dir.path().join("data");
            std::fs::create_dir_all(&data).unwrap();
            let writer = SqliteMetadataWriter::new_with_init(&conn).await.unwrap();
            writer.set_data_path(data.to_str().unwrap()).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("score", DataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from_iter_values(0..ROWS)),
                    Arc::new(Float64Array::from_iter_values((0..ROWS).map(f64::from))),
                ],
            )
            .unwrap();
            DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))
                .unwrap()
                .with_max_row_group_rows(ROW_GROUP)
                .write_table("main", "t", &[batch])
                .await
                .unwrap();
            conn
        }

        /// Plan `sql` (an UPDATE) and return, per source file, its scan's
        /// display and the number of rows the scan yields.
        async fn source_scans(conn: &str, sql: &str) -> Vec<(String, usize)> {
            let provider = SqliteMetadataProvider::new(conn).await.unwrap();
            let writer = SqliteMetadataWriter::new(conn).await.unwrap();
            let catalog =
                DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
            let ctx = SessionContext::new();
            ctx.register_catalog("ducklake", Arc::new(catalog));
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            fn find(
                plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
            ) -> Option<&DuckLakeUpdateExec> {
                plan.downcast_ref::<DuckLakeUpdateExec>()
                    .or_else(|| plan.children().into_iter().find_map(find))
            }
            let update = find(&plan).unwrap_or_else(|| {
                panic!(
                    "no DuckLakeUpdateExec in\n{}",
                    displayable(plan.as_ref()).indent(true)
                )
            });
            let mut out = Vec::new();
            for s in &update.scans {
                let shown = displayable(s.scan.as_ref()).indent(true).to_string();
                let rows = collect(Arc::clone(&s.scan), ctx.task_ctx())
                    .await
                    .unwrap()
                    .iter()
                    .map(|b| b.num_rows())
                    .sum();
                out.push((shown, rows));
            }
            out
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn keyed_update_reads_only_matching_row_groups() {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = seed(&dir).await;
            let scans =
                source_scans(&conn, "UPDATE ducklake.main.t SET score = 0 WHERE id = 22").await;
            assert_eq!(scans.len(), 1);
            let (shown, rows) = &scans[0];
            assert!(
                shown.contains("predicate="),
                "predicate not pushed into the reader:\n{shown}"
            );
            assert!(
                *rows <= ROW_GROUP,
                "scanned {rows} rows of {ROWS}; expected at most one {ROW_GROUP}-row group"
            );
        }

        /// The `DELETE` guard applies: a float predicate is not pushed (footer
        /// bounds exclude NaN), so the whole file is read, as before.
        #[tokio::test(flavor = "multi_thread")]
        async fn float_predicate_is_not_pushed() {
            let dir = tempfile::TempDir::new().unwrap();
            let conn = seed(&dir).await;
            let scans =
                source_scans(&conn, "UPDATE ducklake.main.t SET id = 0 WHERE score = 22").await;
            let (shown, rows) = &scans[0];
            assert!(
                !shown.contains("predicate="),
                "float predicate was pushed:\n{shown}"
            );
            assert_eq!(*rows, ROWS as usize);
        }
    }
}
