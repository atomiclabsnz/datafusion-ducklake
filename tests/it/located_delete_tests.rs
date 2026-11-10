//! `DuckLakeTable::located_rows` and `replace_located_rows` (ADR-0037 slice 5
//! in ferrion): a merge whose matched rows are named by where they live leaves
//! the table `upsert_by_keys` of the same delta leaves, in one snapshot.
//!
//! Each scenario builds the same table twice — two data files, a delete file
//! over one, inlined rows with one of them deleted, NULL ids, and a row two
//! upserts replaced (its old versions in a data file under its delete file) —
//! and merges one delta into each: by `upsert_by_keys` in one, by naming the
//! rows [`located_rows`] finds for the delta's ids and handing them and the
//! delta to `replace_located_rows` in the other. Both must report the same
//! counts, commit one snapshot, and read back the same rows with the same row
//! ids, at the head and at the snapshot before.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::collections::{BTreeMap, BTreeSet, HashSet};

use datafusion_ducklake::roaring::RoaringTreemap;
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

/// Two upserts of id 7 (its first versions left in a data file, deleted): a
/// located scan that returned dead rows would find it three times.
async fn upsert_seven_twice(temp: &TempDir) {
    for s in ["7a", "7b"] {
        let ctx = writable(temp).await;
        let provider = table(&ctx).await;
        let mut delta = rows(&[Some(7)]);
        delta = RecordBatch::try_new(
            delta.schema(),
            vec![Arc::clone(delta.column(0)), Arc::new(StringArray::from(vec![s]))],
        )
        .unwrap();
        as_ducklake(&provider)
            .keyed_mutation(&["id"])
            .upsert_by_keys(&ctx.state(), vec![delta])
            .await
            .unwrap();
    }
}

/// Every object under the lake's data directory.
fn objects(temp: &TempDir) -> BTreeSet<String> {
    fn walk(dir: &std::path::Path, out: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.insert(path.display().to_string());
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(&temp.path().join("data"), &mut out);
    out
}

/// The delta for `ids`: each id with `s` = `new<id>`, and two NULL ids.
fn delta(ids: &BTreeSet<i32>) -> Vec<RecordBatch> {
    let mut all: Vec<Option<i32>> = ids.iter().copied().map(Some).collect();
    all.extend([None, None]);
    let names: Vec<String> = all
        .iter()
        .map(|id| format!("new{}", id.map_or("null".into(), |i| i.to_string())))
        .collect();
    let batch = rows(&all);
    vec![
        RecordBatch::try_new(
            batch.schema(),
            vec![Arc::clone(batch.column(0)), Arc::new(StringArray::from(names))],
        )
        .unwrap(),
    ]
}

fn stream(batches: Vec<RecordBatch>) -> datafusion::execution::SendableRecordBatchStream {
    let schema = batches[0].schema();
    Box::pin(
        datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches.into_iter().map(Ok)),
        ),
    )
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

/// The live rows of `located_rows(["id"])` whose id is in `ids`, as the merge takes them.
async fn locate(
    ctx: &SessionContext,
    table: &DuckLakeTable,
    ids: &BTreeSet<i32>,
) -> (BTreeMap<i64, RoaringTreemap>, HashSet<i64>, usize) {
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
    let mut files: BTreeMap<i64, RoaringTreemap> = BTreeMap::new();
    let mut inlined = HashSet::new();
    let mut sevens = 0;
    for b in &batches {
        let id = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let file = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let row = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            if !id.is_null(i) && id.value(i) == 7 {
                sevens += 1;
            }
            if id.is_null(i) || !ids.contains(&id.value(i)) {
                continue;
            }
            if file.is_null(i) {
                inlined.insert(row.value(i));
            } else {
                files
                    .entry(file.value(i))
                    .or_default()
                    .insert(u64::try_from(row.value(i)).unwrap());
            }
        }
    }
    (files, inlined, sevens)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_located_merge_leaves_the_table_an_upsert_leaves() {
    let all: BTreeSet<i32> = (0..80).chain([100, 101]).collect();
    let scenarios: Vec<(&str, BTreeSet<i32>)> = vec![
        ("one file", [1, 2, 5].into()),
        ("both files and inlined", [0, 39, 40, 79, 100].into()),
        ("a row two upserts replaced", [7, 8].into()),
        ("already deleted, with live", [3, 41, 101, 4].into()),
        ("only already deleted", [3, 41, 101].into()),
        ("no match", [1000, 2000].into()),
        ("past the inlining limit", (10..30).collect()),
        ("every id", all),
    ];
    for (name, ids) in scenarios {
        let by_key = lake().await;
        let by_location = lake().await;
        upsert_seven_twice(&by_key).await;
        upsert_seven_twice(&by_location).await;
        let before = snapshots(&by_key).await;
        assert_eq!(before, snapshots(&by_location).await);
        assert_eq!(read(&by_key, None).await, read(&by_location, None).await);

        let ctx = writable(&by_key).await;
        let provider = table(&ctx).await;
        let by_keys = as_ducklake(&provider)
            .keyed_mutation(&["id"])
            .upsert_by_keys(&ctx.state(), delta(&ids))
            .await
            .unwrap();

        let ctx = writable(&by_location).await;
        let provider = table(&ctx).await;
        let lake_table = as_ducklake(&provider);
        let (files, inlined, sevens) = locate(&ctx, lake_table, &ids).await;
        assert_eq!(
            sevens, 1,
            "{name}: only the live version of id 7 is located"
        );
        let by_located = lake_table
            .replace_located_rows(&ctx.state(), files, inlined, stream(delta(&ids)))
            .await
            .unwrap();

        assert_eq!(
            by_located.rows_deleted, by_keys.rows_deleted,
            "{name}: deleted"
        );
        assert_eq!(
            by_located.rows_inserted, by_keys.rows_inserted,
            "{name}: inserted"
        );
        assert_eq!(
            snapshots(&by_location).await,
            before + 1,
            "{name}: one snapshot"
        );
        assert_eq!(snapshots(&by_key).await, before + 1);
        let head = read(&by_location, None).await;
        assert_eq!(
            head,
            read(&by_key, None).await,
            "{name}: the rows at the head"
        );
        assert_eq!(
            read(&by_location, Some(before)).await,
            read(&by_key, Some(before)).await,
            "{name}: the rows before"
        );
    }
}

/// A location the snapshot does not have live — a data file, a position outside
/// its rows, an inlined row id — is refused before anything is written: no
/// snapshot, and no object left in the lake's data directory, though the delta is
/// past the inlining limit (so it would be written to Parquet).
#[tokio::test(flavor = "multi_thread")]
async fn a_location_the_snapshot_does_not_have_is_refused_before_anything_is_written() {
    let temp = lake().await;
    let before = (
        snapshots(&temp).await,
        read(&temp, None).await,
        objects(&temp),
    );
    let ctx = writable(&temp).await;
    let provider = table(&ctx).await;
    let lake_table = as_ducklake(&provider);
    let ids: BTreeSet<i32> = [1, 100].into_iter().chain(200..220).collect();
    let (files, inlined, _) = locate(&ctx, lake_table, &ids).await;
    assert!(
        !files.is_empty() && !inlined.is_empty(),
        "the fixture's rows"
    );
    let some_file = *files.keys().next().unwrap();

    let mut stale = files.clone();
    stale.insert(987_654, [0u64].into_iter().collect());
    // (A negative position cannot be given: a set holds none.)
    let mut past = files.clone();
    past.get_mut(&some_file).unwrap().insert(1_000_000);
    let mut stale_inlined = inlined.clone();
    stale_inlined.insert(555_555);
    for (what, files, inlined, says) in [
        ("a file that is not live", stale, inlined.clone(), "987654"),
        ("a position past the file", past, inlined.clone(), "1000000"),
        (
            "an inlined row that is not live",
            files.clone(),
            stale_inlined,
            "555555",
        ),
    ] {
        let refused = lake_table
            .replace_located_rows(&ctx.state(), files, inlined, stream(delta(&ids)))
            .await
            .expect_err(what);
        assert!(refused.to_string().contains(says), "{what}: {refused}");
    }
    assert_eq!(
        (
            snapshots(&temp).await,
            read(&temp, None).await,
            objects(&temp)
        ),
        before,
        "nothing was committed or left behind"
    );
}

/// The delete and the append are one commit: a delta that fails after its write
/// to Parquet began (past the inlining limit) leaves the table whole — its
/// matched rows not deleted — with no snapshot and no object left behind.
#[tokio::test(flavor = "multi_thread")]
async fn a_delta_that_fails_mid_stream_leaves_the_table_whole() {
    let temp = lake().await;
    let before = (
        snapshots(&temp).await,
        read(&temp, None).await,
        objects(&temp),
    );
    let ctx = writable(&temp).await;
    let provider = table(&ctx).await;
    let lake_table = as_ducklake(&provider);
    let ids: BTreeSet<i32> = (0..40).collect();
    let (files, inlined, _) = locate(&ctx, lake_table, &ids).await;
    let good = delta(&ids);
    let schema = good[0].schema();
    let failing = Box::pin(
        datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(good.into_iter().map(Ok).chain([Err(
                datafusion::error::DataFusionError::Execution("the delta's source failed".into()),
            )])),
        ),
    );
    let refused = lake_table
        .replace_located_rows(&ctx.state(), files, inlined, failing)
        .await
        .expect_err("the delta failed");
    assert!(refused.to_string().contains("source failed"), "{refused}");
    assert_eq!(
        (
            snapshots(&temp).await,
            read(&temp, None).await,
            objects(&temp)
        ),
        before,
        "the table is whole"
    );
}

/// A delete file of more positions than one of its batches holds (65,536): a merge
/// matching 100,000 rows of one data file deletes exactly them, as `upsert_by_keys` does.
#[tokio::test(flavor = "multi_thread")]
async fn a_delete_file_of_many_batches_deletes_every_position() {
    let build = || async {
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
        let ids: Vec<Option<i32>> = (0..200_000).map(Some).collect();
        writer
            .write_table("main", "t", &[rows(&ids)])
            .await
            .unwrap();
        temp
    };
    let evens: BTreeSet<i32> = (0..200_000).step_by(2).collect();
    let by_key = build().await;
    let by_location = build().await;
    let ctx = writable(&by_key).await;
    let provider = table(&ctx).await;
    as_ducklake(&provider)
        .keyed_mutation(&["id"])
        .upsert_by_keys(&ctx.state(), delta(&evens))
        .await
        .unwrap();
    let ctx = writable(&by_location).await;
    let provider = table(&ctx).await;
    let lake_table = as_ducklake(&provider);
    let (files, inlined, _) = locate(&ctx, lake_table, &evens).await;
    assert_eq!(
        files.values().map(RoaringTreemap::len).sum::<u64>(),
        100_000
    );
    let replaced = lake_table
        .replace_located_rows(&ctx.state(), files, inlined, stream(delta(&evens)))
        .await
        .unwrap();
    assert_eq!(replaced.rows_deleted, 100_000);
    let head = read(&by_location, None).await;
    assert_eq!(head.len(), 200_002);
    assert_eq!(head, read(&by_key, None).await);
}

/// A set of positions naming rows the snapshot already deleted (here every
/// physical row of both data files, ids 3 and 41 deleted by their delete files
/// among them) deletes and counts only the live ones, as a list of positions
/// does: a file whose every live row is named is dropped, not given a delete
/// file that names its deleted rows twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_set_naming_deleted_rows_deletes_only_the_live_ones() {
    let temp = lake().await;
    let ctx = writable(&temp).await;
    let provider = table(&ctx).await;
    let lake_table = as_ducklake(&provider);
    let all: BTreeSet<i32> = (0..80).collect();
    let (located, _, _) = locate(&ctx, lake_table, &all).await;
    assert_eq!(located.len(), 2, "the fixture's two data files");
    let every: BTreeMap<i64, RoaringTreemap> = located
        .keys()
        .map(|file| (*file, (0u64..40).collect()))
        .collect();
    let result = lake_table
        .replace_located_rows(
            &ctx.state(),
            every,
            HashSet::new(),
            stream(delta(&[1000].into())),
        )
        .await
        .unwrap();
    // 40 rows in each file, one of each deleted already.
    assert_eq!(result.rows_deleted, 78);
    assert_eq!(result.rows_inserted, 3);
    let ids: Vec<String> = read(&temp, None)
        .await
        .into_iter()
        .map(|row| row.split(',').skip(1).collect::<Vec<_>>().join(","))
        .collect();
    let mut ids = ids;
    ids.sort();
    assert_eq!(
        ids,
        ["100,s100", "1000,new1000", "NULL,n", "NULL,newnull", "NULL,newnull"],
        "only the inlined rows and the delta are left"
    );
}
