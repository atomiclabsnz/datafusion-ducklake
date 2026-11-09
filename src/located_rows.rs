//! A table's live rows with where each lives, for a keyed merge that names
//! the rows it deletes by a join.
//!
//! A keyed merge whose delta is larger than memory cannot delete by a predicate
//! that holds the delta's keys (`key IN (…)`, or a set of them). It can join
//! the table with the delta instead — a join DataFusion can spill — if the
//! table's rows carry where they live, and then delete the rows the join
//! named and append the delta in one snapshot
//! ([`DuckLakeTable::replace_located_rows`]).
//!
//! [`DuckLakeTable::located_rows`] is a [`TableProvider`] of the table's live
//! rows at its snapshot, each as some of its columns, then
//! [`LOCATED_FILE_COLUMN`] (the catalog id of the data file holding it, NULL
//! for an inlined row) and [`LOCATED_ROW_COLUMN`] (its physical position in
//! that file — a delete file's `pos` — or the inlined row's row id). A data
//! file's rows deleted at the snapshot (by its delete file, or by an inlined
//! deletion) are not among them, so an old version of a row an `UPDATE` or an
//! upsert replaced is never matched.

use std::collections::HashSet;

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

use crate::delete_filter::DeleteFilterExec;
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
        inlined_deletes: Option<&HashSet<i64>>,
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
        // Only the rows live at the snapshot: not those its delete file or an
        // inlined deletion deletes.
        let deleted = self
            .table
            .deleted_positions_at_snapshot(state, file, inlined_deletes)
            .await?;
        let plan: Arc<dyn ExecutionPlan> = if deleted.is_empty() {
            plan
        } else {
            Arc::new(DeleteFilterExec::try_new(
                plan,
                file.file.path.clone(),
                Arc::new(deleted),
                read.len(),
            )?)
        };
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
        let inlined_deletes = self.table.inlined_deletes_by_file()?;
        for file in &files {
            children.push(
                self.file_plan(state, file, inlined_deletes.get(&file.data_file_id))
                    .await?,
            );
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
