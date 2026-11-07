//! A sorted `INSERT` that the fork wrote before its sorts spilled (ADR-0037
//! slice 4 in ferrion) is still written: the room a session is given beside a
//! spilling sort never takes the sort's own (its merge reservation,
//! `sort_spill_reservation_bytes`, 10 MiB by default), and with nothing to
//! spill to none is taken. Under DataFusion's DEFAULT session settings and wide
//! tables, natively and as a tab runs (one partition, a 256 MiB pool, with and
//! without a spill arena), keys sorted by the plan (`c0`) and by the writer
//! (`(c0 + 1)`).
//!
//! Self-contained (its own helpers), so the same file runs on the fork before
//! slice 4: every case here is one it wrote.

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::{FairSpillPool, GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_ducklake::DuckLakeWriteOptions;
use datafusion_ducklake::sort::{NullOrder, SortDirection, SortField};

use crate::inlining_abstraction_tests::{Lake, create_with, sqlite};

const MIB: usize = 1024 * 1024;

/// Run `statement` against `lake`'s SQLite catalog directly.
async fn catalog_sql(lake: &Lake, statement: &str) {
    let url = format!(
        "sqlite:{}?mode=rwc",
        lake.data.parent().unwrap().join("lake.db").display()
    );
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    sqlx::query(sqlx::AssertSqlSafe(statement.to_string()))
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
}

/// `main.t` of `columns` BIGINT columns `c0`…, sorted by `key` (as DuckDB
/// stores it) with `sort_on_insert` on.
async fn wide_table(lake: &Lake, columns: usize, key: &str) {
    let names: Vec<String> = (0..columns).map(|i| format!("c{i}")).collect();
    let defs: Vec<(&str, DataType)> = names
        .iter()
        .map(|name| (name.as_str(), DataType::Int64))
        .collect();
    create_with(lake, "t", &defs).unwrap();
    let table_id = lake.table_id("t");
    let field = SortField {
        sort_key_index: 0,
        expression: "c0".to_string(),
        dialect: "duckdb".to_string(),
        direction: SortDirection::Asc,
        null_order: NullOrder::NullsLast,
    };
    lake.writer.set_sort_spec(table_id, &[field]).unwrap();
    catalog_sql(
        lake,
        &format!(
            "UPDATE ducklake_sort_expression SET expression = '{key}' WHERE table_id = {table_id}"
        ),
    )
    .await;
    lake.writer
        .set_table_setting(table_id, "sort_on_insert", "true")
        .unwrap();
}

/// One setting an insert runs under.
struct Case {
    what: &'static str,
    columns: usize,
    rows: u64,
    pool: usize,
    spills: bool,
    /// One partition, as a tab plans.
    tab: bool,
}

const CASES: [Case; 7] = [
    Case {
        what: "native defaults, 30 columns, 100 rows, 64 MiB, a disk manager",
        columns: 30,
        rows: 100,
        pool: 64 * MIB,
        spills: true,
        tab: false,
    },
    Case {
        what: "native defaults, 30 columns, 100 rows, 64 MiB, no disk manager",
        columns: 30,
        rows: 100,
        pool: 64 * MIB,
        spills: false,
        tab: false,
    },
    Case {
        what: "native defaults, 30 columns, 50,000 rows, 64 MiB, no disk manager",
        columns: 30,
        rows: 50_000,
        pool: 64 * MIB,
        spills: false,
        tab: false,
    },
    Case {
        what: "a tab's worker, 160 columns, 100 rows, 256 MiB, a spill arena",
        columns: 160,
        rows: 100,
        pool: 256 * MIB,
        spills: true,
        tab: true,
    },
    // Measured, 100,000 to 1.6M rows of 3, 8 and 30 columns under 64 MiB with nothing to
    // spill to: the fork before slice 4 and this one refuse the same writes; with the
    // headroom also taken there, these two (and 800,000 rows of 3) were refused too.
    Case {
        what: "a tab's main thread, 30 columns, 100,000 rows, 64 MiB, nothing to spill to",
        columns: 30,
        rows: 100_000,
        pool: 64 * MIB,
        spills: false,
        tab: true,
    },
    Case {
        what: "a tab's main thread, 8 columns, 400,000 rows, 64 MiB, nothing to spill to",
        columns: 8,
        rows: 400_000,
        pool: 64 * MIB,
        spills: false,
        tab: true,
    },
    Case {
        what: "a tab's main thread, 160 columns, 100 rows, 256 MiB, nothing to spill to",
        columns: 160,
        rows: 100,
        pool: 256 * MIB,
        spills: false,
        tab: true,
    },
];

/// Every case, both keys: written, all its rows. On the fork's head before
/// review round 1 of slice 4, the session's headroom (1 MiB a column and a row
/// group, taken whole) left the sort no room for its 10 MiB merge
/// reservation, and the first case was refused at 100 rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_wide_sorted_insert_under_the_default_settings_is_written() {
    let mut failed = Vec::new();
    for case in &CASES {
        for key in ["c0", "(c0 + 1)"] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(case.pool));
            let partitions = case.tab.then_some(1);
            if let Err(e) =
                write_wide(case.columns, case.rows, pool, case.spills, partitions, key).await
            {
                failed.push(format!("{} `{key}`: {e}", case.what));
            }
        }
    }
    assert!(failed.is_empty(), "refused:\n{}", failed.join("\n"));
}

/// `rows` rows of `columns` BIGINTs into a new `main.t` sorted by `key`, under
/// `pool`, spilling to the OS temp directory when `spills`, planned in
/// `partitions` (DataFusion's default, the core count, when `None`) and
/// otherwise at DataFusion's defaults (`sort_spill_reservation_bytes` 10 MiB
/// among them): `Ok` when all of them are written.
async fn write_wide(
    columns: usize,
    rows: u64,
    pool: Arc<dyn MemoryPool>,
    spills: bool,
    partitions: Option<usize>,
    key: &str,
) -> Result<(), String> {
    let lake = sqlite()(None).await;
    wide_table(&lake, columns, key).await;
    let disk = if spills {
        DiskManagerBuilder::default().with_mode(DiskManagerMode::OsTmpDirectory)
    } else {
        DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled)
    };
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .with_disk_manager_builder(disk)
        .build_arc()
        .unwrap();
    let mut config = SessionConfig::new();
    if let Some(partitions) = partitions {
        config = config.with_target_partitions(partitions);
    }
    let ctx = SessionContext::new_with_config_rt(config, runtime);
    let mut options = DuckLakeWriteOptions::default();
    options.data_inlining_row_limit = Some(0);
    ctx.register_catalog("lake", Arc::new(lake.catalog().with_write_options(options)));
    let values: Vec<String> = (0..columns)
        .map(|i| format!("CAST((value * {}) % 1000003 AS BIGINT)", i + 1))
        .collect();
    let sql = format!(
        "INSERT INTO lake.main.t SELECT {} FROM generate_series(1, {rows}) AS g(value)",
        values.join(", ")
    );
    let batches = match ctx.sql(&sql).await {
        Ok(frame) => frame.collect().await.map_err(|e| e.to_string())?,
        Err(e) => return Err(e.to_string()),
    };
    let count = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    if count.contains(&format!("| {rows} ")) {
        Ok(())
    } else {
        Err(format!("answered {count}"))
    }
}

/// The review-round-2 shapes (datafusion-ducklake#35): 100,000 rows of 30
/// BIGINTs, 64 MiB in one partition and 128 MiB in 2, 4 and 16, a disk manager,
/// DataFusion's defaults, under a greedy pool and a fair one (`FairSpillPool`, which
/// splits what is spillable evenly between its spillable consumers), both
/// keys. On the fork's head before round 2, the plan's sort in more than one
/// partition was refused (each partition's sort takes its own merge
/// reservation; the session's room was capped for one), and the writer's sort
/// under a fair pool too (its spillable share in 16 partitions is under one
/// batch's, where the collected path's charge is not shared).
///
/// The fork before slice 4 refuses `c0` in 4 and 16 partitions under the fair
/// pool (each partition's sort's share is under its own batch), and so does
/// this one, so those two are not here; every other one the fork before slice
/// 4 writes (this file passes on it).
#[tokio::test(flavor = "multi_thread")]
async fn a_sorted_insert_in_many_partitions_under_either_pool_is_written() {
    let mut failed = Vec::new();
    for partitions in [1usize, 2, 4, 16] {
        for fair in [false, true] {
            for key in ["c0", "(c0 + 1)"] {
                if partitions >= 4 && fair && key == "c0" {
                    continue;
                }
                // 128 MiB past one partition: at 64 MiB the fork before slice
                // 4 itself refuses `c0` in 2 and 4 partitions on some runs
                // (each partition's sort and the merge above them race for the
                // pool), so no run there tells this head from it.
                let size = if partitions == 1 {
                    64 * MIB
                } else {
                    128 * MIB
                };
                let pool: Arc<dyn MemoryPool> = if fair {
                    Arc::new(FairSpillPool::new(size))
                } else {
                    Arc::new(GreedyMemoryPool::new(size))
                };
                if let Err(e) = write_wide(30, 100_000, pool, true, Some(partitions), key).await {
                    let pool = if fair {
                        "fair"
                    } else {
                        "greedy"
                    };
                    failed.push(format!("{partitions} partitions, {pool} `{key}`: {e}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "refused:\n{}", failed.join("\n"));
}
