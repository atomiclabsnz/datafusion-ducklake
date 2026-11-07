//! DuckLake INSERT execution plan.
//!
//! An unpartitioned table whose rows the plan orders (or that has no sort
//! order) streams its input into a write session, batch by batch: the session
//! holds the first `data_inlining_row_limit` rows to inline, and past them writes
//! Parquet that goes out as it is encoded (ADR-0037 in ferrion).
//!
//! A table the writer itself must sort (an expression key with `sort_on_insert`
//! on, or `sort_on_insert` off while inlining is enabled) streams too, once its
//! input holds more rows than could be inlined: the rows go through DataFusion's
//! spilling sort ([`crate::insert_sort`]) under the task's memory pool and disk
//! manager, then into the session, laid out as the in-memory sort laid them out.
//! A write of at most that many rows is still decided whole, as before.
//!
//! A partitioned table still collects the input first: the rows are split as a
//! whole.
//!
//! Single partition only (partition 0).

use std::fmt::{self, Debug};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::metadata_writer::{MetadataWriter, WriteMode};
use crate::table_writer::DuckLakeTableWriter;

// The resolved write-side partition spec lives in `partition` (it is shared with
// the low-level writer paths and the UPDATE rewrite); re-exported here because
// `DuckLakeInsertExec::new` takes it.
pub use crate::partition::{PartitionWriteKey, PartitionWriteSpec};

/// Schema for the output of insert operations (count of rows inserted)
fn make_insert_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Execution plan that writes input data to a DuckLake table.
pub struct DuckLakeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    writer: Arc<dyn MetadataWriter>,
    schema_name: String,
    table_name: String,
    arrow_schema: SchemaRef,
    write_mode: WriteMode,
    object_store_url: Arc<ObjectStoreUrl>,
    /// When set, the target table is partitioned: input rows are split by the
    /// transformed partition key into one file per partition, all committed in a
    /// single snapshot. `None` for an unpartitioned table (single-file write).
    partition: Option<PartitionWriteSpec>,
    /// Write-layout options applied to the DuckLakeTableWriter built at execute
    /// time (compression, row-group caps, file-rollover target).
    write_options: crate::table_writer::DuckLakeWriteOptions,
    /// The sort order this insert requires of its input: the table's live sort
    /// spec, when the plan sorts by it (`None` leaves any sort to the writer).
    /// Declared via `required_input_ordering` so DataFusion's EnforceSorting
    /// keeps the input sorted instead of pruning the SortExec as unused.
    required_ordering: Option<datafusion::physical_expr::LexOrdering>,
    /// The snapshot the statement was planned at; changes committed since it
    /// conflict with the insert, as official checks its insert.
    planned_snapshot: Option<i64>,
    /// The `sort_on_insert` the statement was planned with, which the writer
    /// lays the rows out under, as official reads it once, in `PlanInsert`.
    /// `None` when the plan read none, and the writer reads it.
    planned_sort_on_insert: Option<bool>,
    /// Set when the statement was planned against a table with no sort order:
    /// the one case, beside a plan that sorted the rows, in which the rows may
    /// stream into the writer. Unset (a plan built through [`Self::new`]
    /// alone) collects them, so the writer can apply a sort order it finds.
    planned_no_sort_order: bool,
    /// What the plan left the writer to do with the rows' order, when the
    /// table has a sort order the plan did not sort by: `None` when it sorted
    /// them, the table has none, or the plan was built through [`Self::new`]
    /// alone (then the rows are collected and the writer reads the order).
    planned_writer_sort: Option<WriterSort>,
    /// The sorts the execution ran ([`crate::insert_sort`]), for
    /// [`ExecutionPlan::metrics`].
    sorts: crate::insert_sort::SortSlot,
    cache: Arc<PlanProperties>,
}

/// What a planned `INSERT` leaves the writer to do with its rows' order
/// ([`DuckLakeInsertExec::with_planned_writer_sort`]).
#[derive(Debug, Clone)]
pub(crate) enum WriterSort {
    /// Nothing: `sort_on_insert` is off and inlining disabled, so official
    /// sorts no row.
    Unsorted,
    /// The rows written to Parquet are sorted by the table's order, as
    /// official sorts them (`sort_on_insert` on with a key the plan left to
    /// the writer, or off while inlining is enabled). An inlined write's rows
    /// are the writer's to order, as before.
    Sorted(crate::sort::SortSpec),
}

impl DuckLakeInsertExec {
    /// Create a new DuckLakeInsertExec
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        writer: Arc<dyn MetadataWriter>,
        schema_name: String,
        table_name: String,
        arrow_schema: SchemaRef,
        write_mode: WriteMode,
        object_store_url: Arc<ObjectStoreUrl>,
        partition: Option<PartitionWriteSpec>,
        write_options: crate::table_writer::DuckLakeWriteOptions,
        required_ordering: Option<datafusion::physical_expr::LexOrdering>,
    ) -> Self {
        let cache = Self::compute_properties();
        Self {
            input,
            writer,
            schema_name,
            table_name,
            arrow_schema,
            write_mode,
            object_store_url,
            partition,
            write_options,
            required_ordering,
            planned_snapshot: None,
            planned_sort_on_insert: None,
            planned_no_sort_order: false,
            planned_writer_sort: None,
            sorts: crate::insert_sort::SortSlot::default(),
            cache,
        }
    }

    fn with_planned_snapshot_opt(mut self, snapshot_id: Option<i64>) -> Self {
        self.planned_snapshot = snapshot_id;
        self
    }

    /// Lay the rows out under `sort_on_insert`, the value the statement was
    /// planned with, rather than reading it again at execution.
    pub(crate) fn with_planned_sort_on_insert(mut self, sort_on_insert: Option<bool>) -> Self {
        self.planned_sort_on_insert = sort_on_insert;
        self
    }

    /// The statement was planned against a table with no sort order, so its rows
    /// may stream into the writer.
    pub(crate) fn with_planned_no_sort_order(mut self, no_sort_order: bool) -> Self {
        self.planned_no_sort_order = no_sort_order;
        self
    }

    /// What the plan left the writer to do with the rows' order
    /// ([`WriterSort`]).
    pub(crate) fn with_planned_writer_sort(mut self, writer_sort: Option<WriterSort>) -> Self {
        self.planned_writer_sort = writer_sort;
        self
    }

    /// Check the insert against the changes committed since `snapshot_id`,
    /// the snapshot the statement was planned at.
    pub(crate) fn with_planned_snapshot(mut self, snapshot_id: i64) -> Self {
        self.planned_snapshot = Some(snapshot_id);
        self
    }

    fn compute_properties() -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(make_insert_count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ))
    }
}

impl Debug for DuckLakeInsertExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DuckLakeInsertExec")
            .field("schema_name", &self.schema_name)
            .field("table_name", &self.table_name)
            .field("write_mode", &self.write_mode)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for DuckLakeInsertExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckLakeInsertExec: schema={}, table={}, mode={:?}",
                    self.schema_name, self.table_name, self.write_mode
                )
            },
        }
    }
}

impl ExecutionPlan for DuckLakeInsertExec {
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn name(&self) -> &str {
        "DuckLakeInsertExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Require all input rows in a single partition.
    ///
    /// `execute` only drives `input.execute(0)`, so without this DataFusion
    /// would feed a multi-partition input (e.g. a parallel scan or aggregation)
    /// straight through and partitions `1..N` would be silently dropped. Asking
    /// for `SinglePartition` makes the optimizer insert a `CoalescePartitionsExec`
    /// that merges every input partition into partition 0 before we read it.
    fn required_input_distribution(&self) -> Vec<datafusion::physical_expr::Distribution> {
        vec![datafusion::physical_expr::Distribution::SinglePartition]
    }

    /// Require the input sorted by the table's live sort order (when one exists),
    /// so rows are laid out sorted within each written file. Declaring it as a hard
    /// requirement stops EnforceSorting from pruning the SortExec `insert_into`
    /// added (a sort with no downstream requirement is otherwise removed).
    fn required_input_ordering(
        &self,
    ) -> Vec<Option<datafusion::physical_expr_common::sort_expr::OrderingRequirements>> {
        vec![self.required_ordering.clone().map(|ordering| {
            datafusion::physical_expr_common::sort_expr::OrderingRequirements::from(ordering)
        })]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(
                "DuckLakeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(
            Self::new(
                Arc::clone(&children[0]),
                Arc::clone(&self.writer),
                self.schema_name.clone(),
                self.table_name.clone(),
                Arc::clone(&self.arrow_schema),
                self.write_mode,
                self.object_store_url.clone(),
                self.partition.clone(),
                self.write_options.clone(),
                self.required_ordering.clone(),
            )
            .with_planned_snapshot_opt(self.planned_snapshot)
            .with_planned_sort_on_insert(self.planned_sort_on_insert)
            .with_planned_no_sort_order(self.planned_no_sort_order)
            .with_planned_writer_sort(self.planned_writer_sort.clone()),
        ))
    }

    /// The metrics of the sorts the writer-sorted path ran (their spills
    /// among them); `None` when it ran none.
    fn metrics(&self) -> Option<datafusion::physical_plan::metrics::MetricsSet> {
        self.sorts.metrics()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "DuckLakeInsertExec only supports partition 0, got {}",
                partition
            )));
        }

        let input = Arc::clone(&self.input);
        let writer = Arc::clone(&self.writer);
        let schema_name = self.schema_name.clone();
        let table_name = self.table_name.clone();
        let arrow_schema = Arc::clone(&self.arrow_schema);
        let write_mode = self.write_mode;
        let object_store_url = self.object_store_url.clone();
        let partition = self.partition.clone();
        let write_options = self.write_options.clone();
        let planned_snapshot = self.planned_snapshot;
        // The plan sorted every row exactly when it required an ordering, so
        // the writer sorts only when it did not, under the plan's setting.
        let sort_plan = crate::table_writer::InsertSortPlan {
            sort_on_insert: self.planned_sort_on_insert,
            sorted: self.required_ordering.is_some(),
        };
        // The rows stream when nothing needs them whole: no partition spec to
        // split them by, and no sort the writer must make. Known, not assumed:
        // the plan sorted them, the statement was planned against a table
        // with no sort order, or the plan found that official sorts none of
        // them. Anything else (a plan built through `new` alone) collects them,
        // and the writer applies whatever sort order it finds.
        let unsorted = sort_plan.sorted
            || self.planned_no_sort_order
            || matches!(self.planned_writer_sort, Some(WriterSort::Unsorted));
        let streams = partition.is_none() && unsorted;
        // ...and a sort the plan left to the writer streams through a spilling
        // sort once the input is past what could be inlined.
        let streamed_sort = match &self.planned_writer_sort {
            Some(WriterSort::Sorted(spec)) if partition.is_none() && !unsorted => {
                Some(spec.clone())
            },
            _ => None,
        };
        let sorts = self.sorts.clone();
        // What the collected path holds: the input, and for a split or a sort
        // a second copy of it.
        let copies = if partition.is_some() || !unsorted {
            2
        } else {
            1
        };
        let output_schema = make_insert_count_schema();

        let stream = stream::once(async move {
            // An encrypted catalog takes no Parquet this crate writes, so its
            // insert is decided on the whole input (inlined, or refused): a
            // session would refuse it at its start, rows it could inline too.
            let encrypted = (streams || streamed_sort.is_some())
                && writer
                    .catalog_encrypted()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let streams = streams && !encrypted;
            let streamed_sort = streamed_sort.filter(|_| !encrypted);
            let mut input_stream = input.execute(0, Arc::clone(&context))?;
            let mut collected =
                datafusion::execution::memory_pool::MemoryConsumer::new("DuckLakeInsertExec")
                    .register(context.memory_pool());
            let batches: Vec<RecordBatch> = if let Some(spec) = streamed_sort {
                let target = StreamedInsert {
                    writer: Arc::clone(&writer),
                    object_store: context
                        .runtime_env()
                        .object_store(object_store_url.as_ref())?,
                    write_options: &write_options,
                    planned_snapshot,
                    schema_name: &schema_name,
                    table_name: &table_name,
                    arrow_schema: &arrow_schema,
                    write_mode,
                };
                // Boxed, as the streamed path's future is.
                match Box::pin(insert_sorted(
                    target,
                    &spec,
                    input_stream,
                    &mut collected,
                    copies,
                    &context,
                    &sorts,
                ))
                .await?
                {
                    Sorted::Written(rows) => {
                        let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![rows]));
                        return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
                    },
                    // At most what could be inlined: decided whole, as before.
                    Sorted::Held(batches) => batches,
                }
            } else if streams {
                // Up to the first batch with rows: an input of none is answered
                // below as before.
                let mut first = None;
                while let Some(batch) = input_stream.next().await {
                    let batch = batch?;
                    if batch.num_rows() > 0 {
                        first = Some(batch);
                        break;
                    }
                }
                match first {
                    Some(first) => {
                        // Boxed: a write session is kilobytes, and this future is
                        // built on the stack of whoever executes the plan.
                        let rows = Box::pin(insert_streamed(
                            StreamedInsert {
                                writer,
                                object_store: context
                                    .runtime_env()
                                    .object_store(object_store_url.as_ref())?,
                                write_options: &write_options,
                                planned_snapshot,
                                schema_name: &schema_name,
                                table_name: &table_name,
                                arrow_schema: &arrow_schema,
                                write_mode,
                            },
                            first,
                            input_stream,
                            context.memory_pool(),
                        ))
                        .await?;
                        let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![rows]));
                        return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
                    },
                    None => Vec::new(),
                }
            } else {
                // Charged to the pool, as the streamed path's session is, so an
                // input past the pool is refused (`ResourcesExhausted`) rather
                // than outgrowing it; held until the write is done.
                let mut batches = Vec::new();
                while let Some(batch) = input_stream.next().await {
                    let batch = batch?;
                    collected.try_grow(batch.get_array_memory_size() * copies)?;
                    batches.push(batch);
                }
                batches
            };

            // An empty input is a genuine no-op only for Append. For
            // Replace/Overwrite we must still run the write session so the prior
            // generation is retired (truncated): finish() registers a 0-row file
            // and finalize_snapshot runs the Replace retirement. Returning early
            // here would leave the old rows live while reporting count=0 success.
            if batches.is_empty() && write_mode == WriteMode::Append {
                let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![0u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
            }

            // Get object store from runtime environment
            let object_store = context
                .runtime_env()
                .object_store(object_store_url.as_ref())?;

            let mut table_writer = DuckLakeTableWriter::new(writer, object_store)
                .map_err(|e| DataFusionError::External(Box::new(e)))?
                .with_options(&write_options);
            if let Some(snapshot_id) = planned_snapshot {
                table_writer = table_writer.with_planned_base_snapshot(snapshot_id);
            }

            let schema_without_metadata =
                Schema::new(arrow_schema.fields().iter().cloned().collect::<Vec<_>>());

            // Partitioned target: split the input into one file per partition and
            // commit them all in one snapshot. An empty input falls through to the
            // single-file path below (so a Replace still retires the prior gen).
            if let Some(spec) = &partition
                && !batches.is_empty()
            {
                let output_schema_ref: SchemaRef = Arc::new(schema_without_metadata.clone());
                let groups = crate::partition::split_batches_by_partition(
                    &output_schema_ref,
                    &batches,
                    spec,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if !groups.is_empty() {
                    let key_names = spec.key_names();
                    let result = table_writer
                        .write_partition_groups_inlined_in(
                            &schema_name,
                            &table_name,
                            &schema_without_metadata,
                            write_mode,
                            spec.partition_id,
                            &key_names,
                            groups,
                            &crate::table_writer::TableWriteOptions::new(),
                            Some(batches.clone()),
                            sort_plan,
                        )
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    let count_array: ArrayRef =
                        Arc::new(UInt64Array::from(vec![result.records_written as u64]));
                    return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
                }
            }

            // Write the rows (sorted by the plan, or by the writer when the
            // plan did not) through the size-rolling writer: it produces one
            // file per target_file_size and commits them in one snapshot. Only
            // for non-empty input — an empty Replace still needs the
            // single-file truncate marker below, and empty Append already
            // returned above.
            //
            // `_unpartitioned_as_planned`: reaching here means the plan resolved the
            // table as unpartitioned. If a SET PARTITIONED BY went live since, the
            // commit fence must reject so the caller re-plans against the new spec —
            // this write must NOT quietly re-lay-out its rows under a spec the plan
            // never saw.
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total_rows > 0 {
                let result = table_writer
                    .write_rows_unpartitioned_as_planned(
                        &schema_name,
                        &table_name,
                        &schema_without_metadata,
                        write_mode,
                        &batches,
                        sort_plan,
                    )
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let count_array: ArrayRef =
                    Arc::new(UInt64Array::from(vec![result.records_written as u64]));
                return Ok(RecordBatch::try_new(output_schema, vec![count_array])?);
            }

            let mut session = table_writer
                .begin_write(
                    &schema_name,
                    &table_name,
                    &schema_without_metadata,
                    write_mode,
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            for batch in &batches {
                session
                    .write_batch(batch)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            }

            let row_count = session.row_count() as u64;

            session
                .finish()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let count_array: ArrayRef = Arc::new(UInt64Array::from(vec![row_count]));
            Ok(RecordBatch::try_new(output_schema, vec![count_array])?)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_insert_count_schema(),
            stream.map_err(|e: DataFusionError| e),
        )))
    }
}

/// Where a streamed insert writes: the table and how its writer is set up.
struct StreamedInsert<'a> {
    writer: Arc<dyn MetadataWriter>,
    object_store: Arc<dyn object_store::ObjectStore>,
    write_options: &'a crate::table_writer::DuckLakeWriteOptions,
    planned_snapshot: Option<i64>,
    schema_name: &'a str,
    table_name: &'a str,
    arrow_schema: &'a SchemaRef,
    write_mode: WriteMode,
}

/// What [`insert_sorted`] did with an insert's rows.
enum Sorted {
    /// Sorted and written: the rows written.
    Written(u64),
    /// Not written: the input, every batch, holding at most the rows that
    /// could be inlined, for the collected path to decide whole.
    Held(Vec<RecordBatch>),
}

/// An insert whose rows the writer sorts by `spec`, streamed: `input`'s
/// batches are held (charged to `held`, `copies` times their size, as the
/// collected path charges them) while they number at most the rows a write
/// could inline ([`DuckLakeTableWriter::inline_row_limit`]); a write of no
/// more is returned [`Sorted::Held`], so its layout is decided whole, as
/// before. Past them no row can be inlined, so the rows are sorted as the
/// writer sorts rows written to Parquet ([`crate::sort::sort_exprs`], in
/// DataFusion's spilling sort over `context`, [`crate::insert_sort`]) and
/// streamed into a session ([`insert_streamed`]). The held batches' charge is
/// released first: the sort charges what it holds.
///
/// A failure of the sort (the pool, the spill budget, the input) before its
/// first row leaves no session; after it, the session is aborted
/// ([`stream_into`]): no catalog row either way, and its uploads removed.
async fn insert_sorted(
    target: StreamedInsert<'_>,
    spec: &crate::sort::SortSpec,
    mut input: SendableRecordBatchStream,
    held: &mut datafusion::execution::memory_pool::MemoryReservation,
    copies: usize,
    context: &Arc<TaskContext>,
    sorts: &crate::insert_sort::SortSlot,
) -> DataFusionResult<Sorted> {
    let external = |e: crate::DuckLakeError| DataFusionError::External(Box::new(e));
    let schema_without_metadata = Schema::new(
        target
            .arrow_schema
            .fields()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    );
    let limit =
        DuckLakeTableWriter::new(Arc::clone(&target.writer), Arc::clone(&target.object_store))
            .map_err(external)?
            .with_options(target.write_options)
            .inline_row_limit(
                target.schema_name,
                target.table_name,
                target.write_mode,
                &schema_without_metadata,
            )
            .map_err(external)?;
    let mut batches = Vec::new();
    let mut rows = 0usize;
    while rows <= limit {
        let Some(batch) = input.next().await else {
            return Ok(Sorted::Held(batches));
        };
        let batch = batch?;
        held.try_grow(batch.get_array_memory_size() * copies)?;
        rows += batch.num_rows();
        batches.push(batch);
    }
    // Room for the session beside the sort: what one row group of these rows
    // takes ([`session_headroom`]), held until the sort's first row. Without
    // it a sort that just fits the pool keeps it all, and the session then
    // finds none (refused at the pool, where with a disk manager the sort
    // would have spilled). Not taken when the pool cannot spare it.
    let estimate = session_headroom(&batches, rows, target.write_options);
    held.free();
    let headroom =
        datafusion::execution::memory_pool::MemoryConsumer::new("DuckLakeInsertExec headroom")
            .register(context.memory_pool());
    if headroom.try_grow(estimate).is_err() {
        tracing::debug!(
            estimate,
            "no headroom for an insert's session beside its sort"
        );
    }
    let schema = input.schema();
    let rest: SendableRecordBatchStream = Box::pin(
        datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            stream::iter(batches.into_iter().map(Ok)).chain(input),
        ),
    );
    let keys = crate::sort::sort_exprs(spec, &schema_without_metadata).map_err(external)?;
    let mut sorted = crate::insert_sort::sort_stream(rest, keys, Arc::clone(context), sorts)?;
    let mut first = None;
    while let Some(batch) = sorted.next().await {
        let batch = batch?;
        if batch.num_rows() > 0 {
            first = Some(batch);
            break;
        }
    }
    let Some(first) = first else {
        return Err(DataFusionError::Internal(
            "an insert's sort returned none of its rows".to_string(),
        ));
    };
    // The session is charged what it holds from here (`stream_into`).
    drop(headroom);
    let rows = Box::pin(insert_streamed(
        target,
        first,
        sorted,
        context.memory_pool(),
    ))
    .await?;
    Ok(Sorted::Written(rows))
}

/// What a write session of rows like `batches` (`rows` of them) holds at
/// most beside a sort: one row group of them buffered (`max_row_group_rows`,
/// DuckDB's 122,880 by default, at most `max_row_group_bytes`) and, for
/// [`Staging::Parts`](crate::Staging::Parts) (`wasm32`'s default), a part.
fn session_headroom(
    batches: &[RecordBatch],
    rows: usize,
    options: &crate::table_writer::DuckLakeWriteOptions,
) -> usize {
    let bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
    let per_row = bytes / rows.max(1);
    let mut group = per_row.saturating_mul(options.max_row_group_rows.unwrap_or(122_880));
    if let Some(cap) = options.max_row_group_bytes {
        group = group.min(cap);
    }
    let part = match options.staging {
        Some(crate::Staging::Parts {
            part_bytes,
        }) => part_bytes,
        // The default on `wasm32`.
        None if cfg!(target_arch = "wasm32") => crate::Staging::DEFAULT_PART_BYTES,
        _ => 0,
    };
    group.saturating_add(part)
}

/// Begin the write session of `target` (refusing a partition spec gone live
/// since planning, as the collected path does) and stream `first` and `rest`
/// into it ([`stream_into`]): the rows written.
async fn insert_streamed(
    target: StreamedInsert<'_>,
    first: RecordBatch,
    rest: SendableRecordBatchStream,
    pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) -> DataFusionResult<u64> {
    let mut table_writer = DuckLakeTableWriter::new(target.writer, target.object_store)
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .with_options(target.write_options);
    if let Some(snapshot_id) = target.planned_snapshot {
        table_writer = table_writer.with_planned_base_snapshot(snapshot_id);
    }
    let schema_without_metadata = Schema::new(
        target
            .arrow_schema
            .fields()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    );
    let session = table_writer
        .begin_write_unpartitioned_as_planned(
            target.schema_name,
            target.table_name,
            &schema_without_metadata,
            target.write_mode,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Box::pin(stream_into(session, first, rest, pool)).await
}

/// Write `first`, then every batch `rest` yields, into `session` as each
/// arrives, and commit: the rows written. What the session holds in memory
/// ([`TableWriteSession::memory_size`](crate::TableWriteSession::memory_size))
/// is charged to `pool` after each batch, the pool the plan's operators use,
/// so one pool bounds the write: past it the write is refused
/// (`ResourcesExhausted`) rather than outgrowing it. A failure of the input,
/// of the write or of the charge aborts the session, which removes what it
/// uploaded and leaves no catalog row, and is returned as it was met.
pub(crate) async fn stream_into(
    mut session: crate::table_writer::TableWriteSession,
    first: RecordBatch,
    mut rest: SendableRecordBatchStream,
    pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) -> DataFusionResult<u64> {
    let held = datafusion::execution::memory_pool::MemoryConsumer::new("DuckLakeInsertExec")
        .register(pool);
    let mut next = Some(Ok(first));
    while let Some(batch) = next {
        let written = match batch {
            // Each of the session's futures boxed, so none sits in this one's
            // state or in its caller's stack frame (a debug build's futures are
            // large, and a plan is executed deep in its caller's stack).
            Ok(batch) => match Box::pin(session.write_batch_async(&batch)).await {
                Ok(()) => held.try_resize(session.memory_size()),
                Err(e) => Err(DataFusionError::External(Box::new(e))),
            },
            Err(e) => Err(e),
        };
        if let Err(e) = written {
            if let Err(abort) = Box::pin(session.abort()).await {
                tracing::warn!(error = %abort, "failed to abort an insert's write session");
            }
            return Err(e);
        }
        next = rest.next().await;
    }
    let rows = session.row_count() as u64;
    Box::pin(session.finish())
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_count_schema() {
        let schema = make_insert_count_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "count");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt64);
    }
}
