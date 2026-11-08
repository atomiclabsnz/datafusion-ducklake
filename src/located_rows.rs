//! A table's rows with where each lives, and a delete of rows named so.
//!
//! A keyed merge whose delta is larger than memory cannot delete by a predicate
//! that holds the delta's keys (`key IN (…)`, or a set of them). It can join
//! the table with the delta instead — a join DataFusion can spill — if the
//! table's rows carry where they live, and then delete the rows the join
//! named. These are those two halves:
//!
//! - [`DuckLakeTable::located_rows`]: a [`TableProvider`] of the table's rows,
//!   each as some of its columns, then [`LOCATED_FILE_COLUMN`] (the catalog id
//!   of the data file holding it, NULL for an inlined row) and
//!   [`LOCATED_ROW_COLUMN`] (its physical position in that file, or the inlined
//!   row's row id). A data file is read raw, delete files NOT applied, as
//!   [`DuckLakeTable::resolve_positions`] reads it for a `DELETE`: the delete
//!   skips a row already deleted, so naming one changes nothing.
//! - [`DuckLakeTable::delete_located_rows`]: the rows so named deleted in one
//!   snapshot, recorded as a SQL `DELETE` over the same rows records them (it
//!   is the same code, with the rows named instead of found by a predicate). A
//!   row named in a file or an inlined table the table's snapshot does not have
//!   live is refused, before anything is written.
//!
//! Both read the table at the snapshot it was opened at, so the rows one names
//! are rows of the snapshot the other deletes from.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, new_null_array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::ScalarValue;
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{CastExpr, Column, IsNotNullExpr, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::union::UnionExec;

use crate::delete_exec::{Selection, run_delete};
use crate::table::DuckLakeTable;

/// The column of [`DuckLakeTable::located_rows`] naming the data file a row is
/// in (its catalog `data_file_id`), NULL for an inlined row.
pub const LOCATED_FILE_COLUMN: &str = "__ducklake_located_file";
/// The column of [`DuckLakeTable::located_rows`] naming the row within its
/// data file (its physical position, a delete file's `pos`), or an inlined
/// row's row id.
pub const LOCATED_ROW_COLUMN: &str = "__ducklake_located_row";

impl DuckLakeTable {
    /// This table's rows with where each lives: `columns` (names of its
    /// columns, in that order), then [`LOCATED_FILE_COLUMN`] and
    /// [`LOCATED_ROW_COLUMN`]. See the [module docs](self).
    ///
    /// Every field is nullable. Refused when a name is not a column of the
    /// table, or is one of the two location columns.
    pub fn located_rows(&self, columns: &[&str]) -> crate::Result<Arc<dyn TableProvider>> {
        let physical = self.physical_schema();
        let mut indices = Vec::with_capacity(columns.len());
        let mut fields = Vec::with_capacity(columns.len() + 2);
        for name in columns {
            if *name == LOCATED_FILE_COLUMN || *name == LOCATED_ROW_COLUMN {
                return Err(crate::DuckLakeError::InvalidConfig(format!(
                    "column `{name}` is a location column of located_rows"
                )));
            }
            let index = physical.index_of(name).map_err(|_| {
                crate::DuckLakeError::InvalidConfig(format!(
                    "'{}' has no column `{name}`",
                    self.table_name()
                ))
            })?;
            indices.push(index);
            fields.push(physical.field(index).clone().with_nullable(true));
        }
        fields.push(Field::new(LOCATED_FILE_COLUMN, DataType::Int64, true));
        fields.push(Field::new(LOCATED_ROW_COLUMN, DataType::Int64, true));
        Ok(Arc::new(LocatedRows {
            table: Arc::new(self.clone()),
            columns: indices,
            schema: Arc::new(Schema::new(fields)),
        }))
    }

    /// Delete, in one snapshot, the rows named by where they live: `files`,
    /// each data file's physical positions (by catalog `data_file_id`), and
    /// `inlined`, inlined rows' row ids — as [`Self::located_rows`] names them.
    /// Returns how many rows were deleted; a row already deleted is not counted
    /// and not deleted again, and when nothing is deleted nothing is committed.
    ///
    /// Recorded as a SQL `DELETE` of the same rows records them, with its
    /// conflict checks: a commit since this table's snapshot that conflicts
    /// aborts it. A data file or an inlined row this table's snapshot does not
    /// have live is refused, and nothing is written. The table must have been
    /// opened from a writable catalog.
    pub async fn delete_located_rows(
        &self,
        state: &dyn Session,
        files: HashMap<i64, Vec<i64>>,
        inlined: HashSet<i64>,
    ) -> DataFusionResult<u64> {
        let writer = self.writer().ok_or_else(|| {
            DataFusionError::Plan(
                "Table is read-only. Use DuckLakeCatalog::with_writer() to enable writes."
                    .to_string(),
            )
        })?;
        let schema_name = self.schema_name().ok_or_else(|| {
            DataFusionError::Internal("Schema name not set for writable table".to_string())
        })?;
        self.write_options()
            .validate()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let session_state = state
            .as_any()
            .downcast_ref::<datafusion::execution::SessionState>()
            .ok_or_else(|| {
                DataFusionError::NotImplemented(
                    "a delete on a DuckLake table requires a DataFusion SessionState session"
                        .to_string(),
                )
            })?;
        if files.is_empty() && inlined.is_empty() {
            return Ok(0);
        }
        run_delete(
            Arc::new(self.clone()),
            session_state,
            Selection::Located {
                files,
                inlined,
            },
            Arc::clone(writer),
            Arc::clone(self.object_store_url()),
            schema_name,
            self.table_name(),
            self.table_id(),
            self.snapshot_id(),
        )
        .await
    }
}

/// [`DuckLakeTable::located_rows`]'s provider.
#[derive(Debug)]
struct LocatedRows {
    table: Arc<DuckLakeTable>,
    /// The catalog indices of the columns it presents, in order.
    columns: Vec<usize>,
    schema: SchemaRef,
}

impl LocatedRows {
    /// `array` as the field at `index` of the schema has it.
    fn conform(&self, array: ArrayRef, index: usize) -> DataFusionResult<ArrayRef> {
        let wanted = self.schema.field(index).data_type();
        if array.data_type() == wanted {
            return Ok(array);
        }
        Ok(arrow::compute::cast(&array, wanted)?)
    }

    /// One data file's rows: its positional scan (delete files not applied),
    /// projected onto the provider's schema.
    async fn file_plan(
        &self,
        state: &dyn Session,
        file: &crate::metadata_provider::DuckLakeTableFile,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let physical = self.table.physical_schema();
        // A predicate over the columns to read, so the positional scan reads them
        // (and only them, where the file's renames allow). It is never evaluated
        // and never handed to the reader (`prune` is false).
        let reads = self
            .columns
            .iter()
            .map(|&index| {
                Arc::new(IsNotNullExpr::new(Arc::new(Column::new(
                    physical.field(index).name(),
                    index,
                )))) as Arc<dyn PhysicalExpr>
            })
            .reduce(|left, right| {
                Arc::new(datafusion::physical_expr::expressions::BinaryExpr::new(
                    left,
                    datafusion::logical_expr::Operator::And,
                    right,
                )) as Arc<dyn PhysicalExpr>
            })
            .unwrap_or_else(|| Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))));
        let (plan, _, read) = self
            .table
            .positional_plan(state, &file.file, &reads, false)
            .await?;
        let input = plan.schema();
        let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::new();
        for (out, &index) in self.columns.iter().enumerate() {
            let rank = read.iter().position(|&c| c == index).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "located_rows: column {index} was not read from data file {}",
                    file.data_file_id
                ))
            })?;
            let field = self.schema.field(out);
            let column: Arc<dyn PhysicalExpr> =
                Arc::new(Column::new(input.field(rank).name(), rank));
            let column = if input.field(rank).data_type() == field.data_type() {
                column
            } else {
                Arc::new(CastExpr::new(column, field.data_type().clone(), None))
            };
            exprs.push((column, field.name().clone()));
        }
        exprs.push((
            Arc::new(Literal::new(ScalarValue::Int64(Some(file.data_file_id)))),
            LOCATED_FILE_COLUMN.to_string(),
        ));
        let position = read.len();
        exprs.push((
            Arc::new(Column::new(input.field(position).name(), position)),
            LOCATED_ROW_COLUMN.to_string(),
        ));
        Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
    }

    /// The table's inlined rows, in the provider's schema.
    fn inlined_batches(&self) -> DataFusionResult<Vec<RecordBatch>> {
        let inlined = self
            .table
            .inlined_data_with_row_ids()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        inlined
            .into_iter()
            .filter(|rows| rows.batch.num_rows() > 0)
            .map(|rows| {
                let n = rows.batch.num_rows();
                let mut arrays = Vec::with_capacity(self.columns.len() + 2);
                for (out, &index) in self.columns.iter().enumerate() {
                    arrays.push(self.conform(Arc::clone(rows.batch.column(index)), out)?);
                }
                arrays.push(new_null_array(&DataType::Int64, n));
                arrays.push(Arc::new(Int64Array::from(rows.row_ids)) as ArrayRef);
                Ok(RecordBatch::try_new(Arc::clone(&self.schema), arrays)?)
            })
            .collect()
    }
}

#[async_trait]
impl TableProvider for LocatedRows {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut children: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
        let files = self
            .table
            .files()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        for file in &files {
            children.push(self.file_plan(state, file).await?);
        }
        let inlined = self.inlined_batches()?;
        if !inlined.is_empty() {
            children.push(MemorySourceConfig::try_new_exec(
                &[inlined],
                Arc::clone(&self.schema),
                None,
            )?);
        }
        let plan: Arc<dyn ExecutionPlan> = match children.len() {
            0 => Arc::new(EmptyExec::new(Arc::clone(&self.schema))),
            1 => children.remove(0),
            _ => UnionExec::try_new(children)?,
        };
        let Some(projection) = projection else {
            return Ok(plan);
        };
        let input = plan.schema();
        let exprs = projection
            .iter()
            .map(|&i| {
                (
                    Arc::new(Column::new(input.field(i).name(), i)) as Arc<dyn PhysicalExpr>,
                    input.field(i).name().clone(),
                )
            })
            .collect::<Vec<_>>();
        Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
    }
}
