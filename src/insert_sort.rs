//! The sort a writer-sorted `INSERT` makes, as DataFusion's spilling `SortExec`
//! under the session's memory pool and disk manager (ADR-0037 slice 4 in ferrion).
//!
//! An `INSERT` into a table with a `SET SORTED BY` order whose plan did not sort
//! the rows (an expression key with `sort_on_insert` on, or `sort_on_insert` off
//! while inlining is enabled) is sorted by the writer. That sort used to be
//! [`crate::table_writer::sort_insert_rows`]: the whole input collected, then a
//! second copy of it sorted in memory. Here the same sort is a stream: the rows
//! go through a `SortExec` built at execution, over the task's
//! [`TaskContext`], so they spill wherever the session spills (a disk manager
//! natively, the OPFS arena in a browser worker) and are refused at the pool
//! where it has none.
//!
//! It lays the rows out exactly as `sort_insert_rows` does:
//!
//! - **The same keys**: [`crate::sort::sort_exprs`], so a float key orders as
//!   DuckDB orders it and an expression key is DataFusion's.
//! - **The same order**: each key's values, then the row's arrival position,
//!   which `sort_batches_by_spec` breaks ties with. With the position the order
//!   is total, so neither the in-memory sort's nor the merge's tie rules can
//!   change it.
//! - **The same failure rule**: a key that fails to evaluate over the rows (a
//!   division by zero, a failing cast) leaves every row in arrival order rather
//!   than failing the write. The keys are evaluated as columns before the sort
//!   (which also makes the sorter count them, ferrion's `SortKeysAsColumns`
//!   point), so a key's failure is known for what it is: it is recorded, every
//!   later batch carries null keys, and once the input is read the rows are
//!   sorted again by position alone. Any other error (the pool, the spill
//!   budget, the input) fails the write.
//! - **The same batches**: the sorted rows are cut back to the lengths the input
//!   arrived in, as [`crate::table_writer::reslice_to_lengths`] cuts them, so a
//!   file rolls where it rolled.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{ArrayRef, RecordBatch, UInt64Array, new_null_array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use futures::{StreamExt, TryStreamExt, stream};

/// The sorts an insert ran, kept for its metrics
/// ([`DuckLakeInsertExec`](crate::insert_exec::DuckLakeInsertExec)'s
/// `metrics`, which `EXPLAIN ANALYZE` shows): their spills are counted there.
#[derive(Debug, Clone, Default)]
pub(crate) struct SortSlot(Arc<Mutex<Vec<Arc<dyn ExecutionPlan>>>>);

impl SortSlot {
    fn push(&self, sort: Arc<dyn ExecutionPlan>) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).push(sort);
    }

    /// Every metric of every sort run so far; `None` when none ran.
    pub(crate) fn metrics(&self) -> Option<MetricsSet> {
        let sorts = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if sorts.is_empty() {
            return None;
        }
        let mut set = MetricsSet::new();
        for metric in sorts.iter().filter_map(|sort| sort.metrics()) {
            for metric in metric.iter() {
                set.push(Arc::clone(metric));
            }
        }
        Some(set)
    }
}

/// The name of the appended column holding key `index`'s values.
fn key_column(index: usize) -> String {
    format!("__ducklake_insert_sort_key_{index}")
}

/// The name of the appended column holding each row's arrival position.
const POSITION_COLUMN: &str = "__ducklake_insert_sort_position";

/// `input` in the order of `keys` (each evaluated over the input's rows), ties in
/// arrival order, cut back to the input's batch lengths: what
/// [`crate::table_writer::sort_insert_rows`] returns, as a stream that spills
/// (see the module). `slot` keeps the sorts for their metrics.
///
/// `keys` empty, or a key whose type cannot be planned over the input, returns
/// `input` as it is: the rows keep their arrival order, as
/// `sort_batches_by_spec` keeps them.
pub(crate) fn sort_stream(
    input: SendableRecordBatchStream,
    keys: Vec<PhysicalSortExpr>,
    context: Arc<TaskContext>,
    slot: &SortSlot,
) -> Result<SendableRecordBatchStream> {
    let input_schema = input.schema();
    let Ok(key_types) = keys
        .iter()
        .map(|key| key.expr.data_type(&input_schema))
        .collect::<Result<Vec<DataType>>>()
    else {
        return Ok(input);
    };
    if keys.is_empty() {
        return Ok(input);
    }
    let width = input_schema.fields().len();
    let mut fields: Vec<Field> = input_schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    for (index, data_type) in key_types.iter().enumerate() {
        fields.push(Field::new(key_column(index), data_type.clone(), true));
    }
    fields.push(Field::new(POSITION_COLUMN, DataType::UInt64, false));
    let augmented: SchemaRef = Arc::new(Schema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    ));
    let position = width + keys.len();

    let mut ordering: Vec<PhysicalSortExpr> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            PhysicalSortExpr::new(
                Arc::new(Column::new(&key_column(index), width + index)),
                key.options,
            )
        })
        .collect();

    let failed = Arc::new(AtomicBool::new(false));
    let lengths = Arc::new(Mutex::new(VecDeque::new()));
    let evaluated = {
        let augmented = Arc::clone(&augmented);
        let failed = Arc::clone(&failed);
        let lengths = Arc::clone(&lengths);
        let mut next_position = 0u64;
        input.map(move |batch| {
            let batch = batch?;
            let rows = batch.num_rows();
            lengths
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push_back(rows);
            let mut columns = batch.columns().to_vec();
            let mut values: Vec<ArrayRef> = Vec::with_capacity(keys.len());
            if !failed.load(Ordering::Relaxed) {
                for (key, data_type) in keys.iter().zip(&key_types) {
                    match key.expr.evaluate(&batch).and_then(|v| v.into_array(rows)) {
                        Ok(array) if array.data_type() == data_type => values.push(array),
                        // The key's failure, not the write's: every row keeps its
                        // arrival order (below), as `sort_batches_by_spec` keeps it.
                        _ => {
                            failed.store(true, Ordering::Relaxed);
                            break;
                        },
                    }
                }
            }
            if values.len() != keys.len() {
                values = key_types
                    .iter()
                    .map(|data_type| new_null_array(data_type, rows))
                    .collect();
            }
            columns.extend(values);
            columns.push(Arc::new(UInt64Array::from_iter_values(
                next_position..next_position + rows as u64,
            )));
            next_position += rows as u64;
            Ok(RecordBatch::try_new(Arc::clone(&augmented), columns)?)
        })
    };

    ordering.push(PhysicalSortExpr::new_default(Arc::new(Column::new(
        POSITION_COLUMN,
        position,
    ))));
    let sorted = sort_by(
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&augmented),
            evaluated,
        )),
        ordering,
        Arc::clone(&context),
        slot,
    )?;

    // The sort is blocking: its first batch comes once every input row is read,
    // so by then a key's failure is known.
    let slot = slot.clone();
    let ordered = {
        let augmented = Arc::clone(&augmented);
        stream::once(async move {
            let mut sorted = sorted;
            let first = sorted.next().await;
            let head = Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&augmented),
                stream::iter(first).chain(sorted),
            ));
            if !failed.load(Ordering::Relaxed) {
                return Ok(head as SendableRecordBatchStream);
            }
            let by_position = vec![PhysicalSortExpr::new_default(Arc::new(Column::new(
                POSITION_COLUMN,
                position,
            )))];
            sort_by(head, by_position, context, &slot)
        })
        .try_flatten()
    };

    let projected = {
        let input_schema = Arc::clone(&input_schema);
        ordered.map(move |batch| {
            let batch = batch?;
            Ok(RecordBatch::try_new(
                Arc::clone(&input_schema),
                batch.columns()[..width].to_vec(),
            )?)
        })
    };
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        Arc::clone(&input_schema),
        cut_to_lengths(
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&input_schema),
                projected,
            )),
            lengths,
        ),
    )))
}

/// `input` sorted by `ordering` in a `SortExec` over `context`, kept in `slot`.
fn sort_by(
    input: SendableRecordBatchStream,
    ordering: Vec<PhysicalSortExpr>,
    context: Arc<TaskContext>,
    slot: &SortSlot,
) -> Result<SendableRecordBatchStream> {
    let ordering = LexOrdering::new(ordering)
        .ok_or_else(|| DataFusionError::Internal("an insert's sort has no key".to_string()))?;
    let source = StreamingTableExec::try_new(
        input.schema(),
        vec![Arc::new(OnceStream::new(input)) as Arc<dyn PartitionStream>],
        None,
        Vec::<LexOrdering>::new(),
        false,
        None,
    )?;
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(ordering, Arc::new(source)));
    slot.push(Arc::clone(&sort));
    sort.execute(0, context)
}

/// `input`'s rows, in its order, as batches of the lengths `lengths` holds (an
/// empty one skipped), as [`crate::table_writer::reslice_to_lengths`] cuts
/// them. The lengths are read only once `input` has yielded: they are the
/// sort's input's, all known by its first output. Rows past the lengths (none,
/// unless the sort's rows and its input's disagree) are yielded as they come,
/// rather than dropped.
///
/// Holds at most one length's worth of rows beyond the batch being cut.
fn cut_to_lengths(
    input: SendableRecordBatchStream,
    lengths: Arc<Mutex<VecDeque<usize>>>,
) -> impl futures::Stream<Item = Result<RecordBatch>> + Send {
    struct Cut {
        input: SendableRecordBatchStream,
        lengths: Arc<Mutex<VecDeque<usize>>>,
        want: Option<usize>,
        pending: VecDeque<RecordBatch>,
        pending_rows: usize,
        done: bool,
    }

    impl Cut {
        fn next_length(&self) -> Option<usize> {
            let mut lengths = self.lengths.lock().unwrap_or_else(|p| p.into_inner());
            while let Some(length) = lengths.pop_front() {
                if length > 0 {
                    return Some(length);
                }
            }
            None
        }

        /// The first `rows` pending rows, as one batch.
        fn take(&mut self, rows: usize) -> Result<RecordBatch> {
            let mut pieces = Vec::new();
            let mut taken = 0;
            while taken < rows {
                let Some(batch) = self.pending.pop_front() else {
                    break;
                };
                let need = rows - taken;
                if batch.num_rows() > need {
                    pieces.push(batch.slice(0, need));
                    self.pending
                        .push_front(batch.slice(need, batch.num_rows() - need));
                    taken += need;
                } else {
                    taken += batch.num_rows();
                    pieces.push(batch);
                }
            }
            self.pending_rows -= taken;
            match pieces.len() {
                1 => Ok(pieces.remove(0)),
                _ => Ok(arrow::compute::concat_batches(
                    &self.input.schema(),
                    &pieces,
                )?),
            }
        }
    }

    let cut = Cut {
        input,
        lengths,
        want: None,
        pending: VecDeque::new(),
        pending_rows: 0,
        done: false,
    };
    stream::unfold(cut, |mut cut| async move {
        loop {
            if cut.want.is_none() && (cut.pending_rows > 0 || cut.done) {
                cut.want = cut.next_length();
            }
            if let Some(want) = cut.want
                && cut.pending_rows >= want
            {
                cut.want = None;
                let batch = cut.take(want);
                return Some((batch, cut));
            }
            if cut.done {
                if cut.pending_rows == 0 {
                    return None;
                }
                let rows = cut.pending_rows;
                let batch = cut.take(rows);
                return Some((batch, cut));
            }
            match cut.input.next().await {
                Some(Ok(batch)) => {
                    if batch.num_rows() > 0 {
                        cut.pending_rows += batch.num_rows();
                        cut.pending.push_back(batch);
                    }
                },
                Some(Err(error)) => {
                    cut.done = true;
                    cut.pending.clear();
                    cut.pending_rows = 0;
                    return Some((Err(error), cut));
                },
                None => cut.done = true,
            }
        }
    })
}

/// One stream, as the single partition of a [`StreamingTableExec`], so a
/// `SortExec` can be built over a stream already running.
struct OnceStream {
    schema: SchemaRef,
    stream: Mutex<Option<SendableRecordBatchStream>>,
}

impl OnceStream {
    fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            schema: stream.schema(),
            stream: Mutex::new(Some(stream)),
        }
    }
}

impl std::fmt::Debug for OnceStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnceStream").finish_non_exhaustive()
    }
}

impl PartitionStream for OnceStream {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        match self.stream.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(stream) => stream,
            None => Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream::once(async {
                    Err(DataFusionError::Internal(
                        "an insert's sort input was executed twice".to_string(),
                    ))
                }),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::compute::SortOptions;
    use datafusion::execution::memory_pool::FairSpillPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::SessionConfig;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, false),
        ]))
    }

    /// `rows` rows in batches of `per_batch`: `k` a scrambled key with ties and
    /// NULLs, `s` the row's arrival position as text, so a tie's order shows.
    fn batches(rows: i64, per_batch: i64) -> Vec<RecordBatch> {
        (0..rows)
            .step_by(per_batch as usize)
            .map(|start| {
                let range = start..(start + per_batch).min(rows);
                let k: Int64Array = range
                    .clone()
                    .map(|i| (i % 13 != 0).then_some((i * 7919) % 997))
                    .collect();
                let s: StringArray = range.map(|i| Some(format!("{i:08}"))).collect();
                RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(s)]).unwrap()
            })
            .collect()
    }

    fn stream_of(batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            schema(),
            stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    fn key(options: SortOptions) -> Vec<PhysicalSortExpr> {
        vec![PhysicalSortExpr::new(Arc::new(Column::new("k", 0)), options)]
    }

    fn spilling_context(pool_bytes: usize) -> Arc<TaskContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(pool_bytes)))
            .build_arc()
            .unwrap();
        Arc::new(
            TaskContext::default()
                .with_session_config(
                    SessionConfig::new()
                        .with_batch_size(1000)
                        .with_sort_spill_reservation_bytes(64 * 1024),
                )
                .with_runtime(runtime),
        )
    }

    /// What `sort_insert_rows` returns for the same rows: the in-memory sort.
    fn in_memory(batches: Vec<RecordBatch>, options: SortOptions) -> Vec<RecordBatch> {
        let lengths: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        let combined = arrow::compute::concat_batches(&schema(), &batches).unwrap();
        let rows = combined.num_rows();
        let indices = arrow::compute::lexsort_to_indices(
            &[
                arrow::compute::SortColumn {
                    values: Arc::clone(combined.column(0)),
                    options: Some(options),
                },
                arrow::compute::SortColumn {
                    values: Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
                    options: None,
                },
            ],
            None,
        )
        .unwrap();
        let columns = combined
            .columns()
            .iter()
            .map(|c| arrow::compute::take(c, &indices, None).unwrap())
            .collect();
        let sorted = RecordBatch::try_new(schema(), columns).unwrap();
        crate::table_writer::reslice_to_lengths(vec![sorted], &lengths)
    }

    async fn run(
        batches: Vec<RecordBatch>,
        keys: Vec<PhysicalSortExpr>,
        context: Arc<TaskContext>,
    ) -> (Vec<RecordBatch>, SortSlot) {
        let slot = SortSlot::default();
        let sorted = sort_stream(stream_of(batches), keys, context, &slot).unwrap();
        (sorted.try_collect().await.unwrap(), slot)
    }

    fn spills(slot: &SortSlot) -> usize {
        slot.metrics().and_then(|m| m.spill_count()).unwrap_or(0)
    }

    /// Spilled or not, each direction and null order lays the rows out as the
    /// in-memory sort does, in its batch lengths; and under a small pool with a
    /// disk manager it does spill.
    #[tokio::test]
    async fn the_spilling_sort_is_the_in_memory_sort() {
        let input = batches(60_000, 777);
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let options = SortOptions {
                    descending,
                    nulls_first,
                };
                let expected = in_memory(input.clone(), options);
                let (fits, slot) = run(
                    input.clone(),
                    key(options),
                    Arc::new(TaskContext::default()),
                )
                .await;
                assert_eq!(fits, expected, "{options:?}, in memory");
                assert_eq!(spills(&slot), 0, "{options:?}: nothing spilled");
                let (spilled, slot) =
                    run(input.clone(), key(options), spilling_context(1_000_000)).await;
                assert_eq!(spilled, expected, "{options:?}, spilled");
                assert!(spills(&slot) > 0, "{options:?}: the sort spilled");
            }
        }
    }

    /// A key that fails over the rows (here on the last batch only) leaves every
    /// row in arrival order, spilled or not.
    #[tokio::test]
    async fn a_failing_key_keeps_arrival_order() {
        // `k` is its row's position plus one, except one row of the last batch,
        // whose 0 makes `1 / k` fail there and nowhere else.
        let zero_at = 19_990i64;
        let input: Vec<RecordBatch> = (0..20_000i64)
            .step_by(1000)
            .map(|start| {
                let k: Int64Array = (start..start + 1000)
                    .map(|i| {
                        Some(if i == zero_at {
                            0
                        } else {
                            i + 1
                        })
                    })
                    .collect();
                let s: StringArray = (start..start + 1000)
                    .map(|i| Some(format!("{i:08}")))
                    .collect();
                RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(s)]).unwrap()
            })
            .collect();
        let failing = datafusion::physical_expr::expressions::binary(
            datafusion::physical_expr::expressions::lit(1i64),
            datafusion::logical_expr::Operator::Divide,
            Arc::new(Column::new("k", 0)),
            &schema(),
        )
        .unwrap();
        let keys = vec![PhysicalSortExpr::new(failing, SortOptions::default())];
        for context in [Arc::new(TaskContext::default()), spilling_context(1_000_000)] {
            let (out, _) = run(input.clone(), keys.clone(), context).await;
            assert_eq!(out, input, "arrival order, in the input's batches");
        }
    }

    /// An error of the input is the write's: it is returned, not swallowed.
    #[tokio::test]
    async fn an_input_error_fails_the_sort() {
        let good = batches(100, 50);
        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema(),
            stream::iter(vec![
                Ok(good[0].clone()),
                Err(DataFusionError::Execution("the input failed".to_string())),
            ]),
        ));
        let slot = SortSlot::default();
        let sorted = sort_stream(
            input,
            key(SortOptions::default()),
            Arc::new(TaskContext::default()),
            &slot,
        )
        .unwrap();
        let error = sorted.try_collect::<Vec<_>>().await.unwrap_err();
        assert!(error.to_string().contains("the input failed"), "{error}");
    }

    /// Past the pool with no disk manager, the sort is refused at the pool.
    #[tokio::test]
    async fn past_the_pool_without_a_disk_manager_is_refused() {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(200_000)))
            .with_disk_manager_builder(
                datafusion::execution::disk_manager::DiskManagerBuilder::default()
                    .with_mode(datafusion::execution::disk_manager::DiskManagerMode::Disabled),
            )
            .build_arc()
            .unwrap();
        let context = Arc::new(TaskContext::default().with_runtime(runtime));
        let slot = SortSlot::default();
        let sorted = sort_stream(
            stream_of(batches(60_000, 777)),
            key(SortOptions::default()),
            context,
            &slot,
        )
        .unwrap();
        let error = sorted.try_collect::<Vec<_>>().await.unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
    }
}
