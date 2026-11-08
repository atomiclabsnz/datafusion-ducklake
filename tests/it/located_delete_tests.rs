//! `DuckLakeTable::located_rows` and `delete_located_rows` (ADR-0037 slice 5 in
//! ferrion): a delete of rows named by where they live leaves the table a SQL
//! `DELETE` of the same rows leaves.
//!
//! Each scenario builds the same table twice — two data files, a delete file
//! over one, inlined rows with one of them deleted, NULL ids — and deletes one
//! set of ids from each: by `DELETE … WHERE id IN (…)` in one, by naming the
//! rows [`located_rows`] finds for those ids in the other. Both must report the
//! same count, commit the same number of snapshots, and read back the same rows
//! with the same row ids, at the head and at the snapshot before.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, LOCATED_FILE_COLUMN, LOCATED_ROW_COLUMN,
    MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
};
use sqlx::sqlite::SqlitePool;

fn url(temp: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", temp.path().join("test.db").display())
}

fn rows(ids: &[Option<i32>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let s: Vec<Option<String>> = ids
        .iter()
        .map(|id| Some(format!("s{}", id.unwrap_or(-1))))
        .collect();
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids.to_vec())), Arc::new(StringArray::from(s))],
    )
    .unwrap()
}

async fn writable(temp: &TempDir) -> SessionContext {
    let writer = SqliteMetadataWriter::new(&url(temp))
        .await
        .unwrap()
        .with_data_inlining_row_limit(5);
    let provider = SqliteMetadataProvider::new(&url(temp)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    ctx
}

/// Two data files (ids 0..40 with every seventh NULL, then 40..80), a delete
/// file over the first (ids 3 and 41 deleted), and three inlined rows (100, 101,
/// NULL) of which 101 is deleted.
async fn lake() -> TempDir {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let writer = SqliteMetadataWriter::new_with_init(&url(&temp))
        .await
        .unwrap()
        .with_data_inlining_row_limit(0);
    writer.set_data_path(data.to_str().unwrap()).unwrap();
    let writer =
        DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new())).unwrap();
    let first: Vec<Option<i32>> = (0..40).map(|i| (i % 7 != 6).then_some(i)).collect();
    writer
        .write_table("main", "t", &[rows(&first)])
        .await
        .unwrap();
    let second: Vec<Option<i32>> = (40..80).map(Some).collect();
    writer
        .append_table("main", "t", &[rows(&second)])
        .await
        .unwrap();
    let ctx = writable(&temp).await;
    for sql in [
        "DELETE FROM ducklake.main.t WHERE id = 3 OR id = 41",
        "INSERT INTO ducklake.main.t VALUES (100, 's100'), (101, 's101'), (NULL, 'n')",
        "DELETE FROM ducklake.main.t WHERE id = 101",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    temp
}

async fn table(ctx: &SessionContext) -> Arc<dyn TableProvider> {
    ctx.catalog("ducklake")
        .unwrap()
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap()
}

fn as_ducklake(table: &Arc<dyn TableProvider>) -> &DuckLakeTable {
    (table.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("a DuckLakeTable")
}

/// Every row as `rowid,id,s`, ordered, at the head or at `version`.
async fn read(temp: &TempDir, version: Option<i64>) -> Vec<String> {
    let provider = SqliteMetadataProvider::new(&url(temp)).await.unwrap();
    let catalog = match version {
        Some(v) => DuckLakeCatalog::with_snapshot(Arc::new(provider), v).unwrap(),
        None => DuckLakeCatalog::new(provider).unwrap(),
    }
    .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT rowid, id, s FROM ducklake.main.t ORDER BY rowid")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let rowid = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let id = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let s = arrow::compute::cast(b.column(2), &DataType::Utf8).unwrap();
        let s = s.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push(format!(
                "{},{},{}",
                rowid.value(i),
                if id.is_null(i) {
                    "NULL".into()
                } else {
                    id.value(i).to_string()
                },
                s.value(i)
            ));
        }
    }
    out
}

async fn snapshots(temp: &TempDir) -> i64 {
    let pool = SqlitePool::connect(&url(temp)).await.unwrap();
    let n = sqlx::query_scalar("SELECT max(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    pool.close().await;
    n
}

/// The rows of `located_rows(["id"])` whose id is in `ids`, as the delete takes them.
async fn locate(
    ctx: &SessionContext,
    table: &DuckLakeTable,
    ids: &BTreeSet<i32>,
) -> (HashMap<i64, Vec<i64>>, HashSet<i64>) {
    let located = table.located_rows(&["id"]).unwrap();
    let schema = located.schema();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["id", LOCATED_FILE_COLUMN, LOCATED_ROW_COLUMN]
    );
    let batches = ctx.read_table(located).unwrap().collect().await.unwrap();
    let mut files: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut inlined = HashSet::new();
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let file = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let row = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            if id.is_null(i) || !ids.contains(&id.value(i)) {
                continue;
            }
            if file.is_null(i) {
                inlined.insert(row.value(i));
            } else {
                files.entry(file.value(i)).or_default().push(row.value(i));
            }
        }
    }
    (files, inlined)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_located_delete_leaves_the_table_a_sql_delete_leaves() {
    let all: BTreeSet<i32> = (0..80).chain([100, 101]).collect();
    let scenarios: Vec<(&str, BTreeSet<i32>)> = vec![
        ("one file", [1, 2, 5].into()),
        ("both files and inlined", [0, 39, 40, 79, 100].into()),
        ("already deleted, with live", [3, 41, 101, 4].into()),
        ("only already deleted", [3, 41, 101].into()),
        ("no match", [1000, 2000].into()),
        ("every id", all),
    ];
    for (name, ids) in scenarios {
        let by_sql = lake().await;
        let by_location = lake().await;
        let before = snapshots(&by_sql).await;
        assert_eq!(before, snapshots(&by_location).await);
        assert_eq!(read(&by_sql, None).await, read(&by_location, None).await);

        let list = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let ctx = writable(&by_sql).await;
        let counted = ctx
            .sql(&format!("DELETE FROM ducklake.main.t WHERE id IN ({list})"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let sql_count = counted[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap()
            .value(0);

        let ctx = writable(&by_location).await;
        let provider = table(&ctx).await;
        let lake_table = as_ducklake(&provider);
        let (files, inlined) = locate(&ctx, lake_table, &ids).await;
        let located_count = lake_table
            .delete_located_rows(&ctx.state(), files, inlined)
            .await
            .unwrap();

        assert_eq!(located_count, sql_count, "{name}: the count");
        assert_eq!(
            snapshots(&by_location).await,
            snapshots(&by_sql).await,
            "{name}: the snapshots committed"
        );
        let head = read(&by_location, None).await;
        assert_eq!(
            head,
            read(&by_sql, None).await,
            "{name}: the rows at the head"
        );
        assert_eq!(
            read(&by_location, Some(before)).await,
            read(&by_sql, Some(before)).await,
            "{name}: the rows before"
        );
        if name == "every id" {
            assert_eq!(
                head.len(),
                6,
                "{name}: only the NULL ids are left: {head:?}"
            );
        }
    }
}

/// A row named in a data file or an inlined table the snapshot does not have live
/// is refused, and nothing is committed — not even the rows named rightly beside it.
#[tokio::test(flavor = "multi_thread")]
async fn a_row_the_snapshot_does_not_have_is_refused_and_nothing_is_deleted() {
    let temp = lake().await;
    let before = (snapshots(&temp).await, read(&temp, None).await);
    let ctx = writable(&temp).await;
    let provider = table(&ctx).await;
    let lake_table = as_ducklake(&provider);
    let (mut files, inlined) = locate(&ctx, lake_table, &[1, 100].into()).await;
    assert!(
        !files.is_empty() && !inlined.is_empty(),
        "the fixture's rows"
    );

    files.insert(987_654, vec![0]);
    let refused = lake_table
        .delete_located_rows(&ctx.state(), files.clone(), inlined.clone())
        .await
        .expect_err("a file that is not live");
    assert!(refused.to_string().contains("987654"), "{refused}");
    files.remove(&987_654);

    let mut stale = inlined.clone();
    stale.insert(555_555);
    let refused = lake_table
        .delete_located_rows(&ctx.state(), files, stale)
        .await
        .expect_err("an inlined row that is not live");
    assert!(refused.to_string().contains("555555"), "{refused}");

    assert_eq!(
        (snapshots(&temp).await, read(&temp, None).await),
        before,
        "nothing was committed"
    );
}
