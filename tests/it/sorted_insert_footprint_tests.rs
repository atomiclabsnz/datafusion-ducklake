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
use datafusion::execution::memory_pool::GreedyMemoryPool;
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
            let lake = sqlite()(None).await;
            wide_table(&lake, case.columns, key).await;
            let disk = if case.spills {
                DiskManagerBuilder::default().with_mode(DiskManagerMode::OsTmpDirectory)
            } else {
                DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled)
            };
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::new(GreedyMemoryPool::new(case.pool)))
                .with_disk_manager_builder(disk)
                .build_arc()
                .unwrap();
            // DataFusion's defaults, `sort_spill_reservation_bytes` (10 MiB)
            // among them; a tab plans one partition.
            let mut config = SessionConfig::new();
            if case.tab {
                config = config.with_target_partitions(1);
            }
            let ctx = SessionContext::new_with_config_rt(config, runtime);
            let mut options = DuckLakeWriteOptions::default();
            options.data_inlining_row_limit = Some(0);
            ctx.register_catalog("lake", Arc::new(lake.catalog().with_write_options(options)));
            let values: Vec<String> = (0..case.columns)
                .map(|i| format!("CAST((value * {}) % 1000003 AS BIGINT)", i + 1))
                .collect();
            let sql = format!(
                "INSERT INTO lake.main.t SELECT {} FROM generate_series(1, {}) AS g(value)",
                values.join(", "),
                case.rows
            );
            let result = match ctx.sql(&sql).await {
                Ok(frame) => frame.collect().await.map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            match result {
                Ok(batches) => {
                    let count = arrow::util::pretty::pretty_format_batches(&batches)
                        .unwrap()
                        .to_string();
                    if !count.contains(&format!("| {} ", case.rows)) {
                        failed.push(format!("{} `{key}`: answered {count}", case.what));
                    }
                },
                Err(e) => failed.push(format!("{} `{key}`: {e}", case.what)),
            }
        }
    }
    assert!(failed.is_empty(), "refused:\n{}", failed.join("\n"));
}
