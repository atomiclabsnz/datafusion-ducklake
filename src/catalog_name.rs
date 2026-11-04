//! How a schema, table or view name is compared: DuckDB's rule, in one place per SQL dialect.
//!
//! DuckDB stores a catalog name as written (quoted or not, and a `CREATE OR REPLACE` stores its
//! new spelling) and finds it in any **ASCII** case: `lake.main.orders` names a table created as
//! `"Orders"`, and two live names of one namespace may not differ only by ASCII case, whatever
//! their kinds (a table `sales` beside a view `"Sales"` is refused). `"Ärger"` and `"ärger"` are
//! two names. `tests/it/name_case_tests.rs` pins this against DuckDB 1.5.6
//! (atomiclabsnz/ferrion#708, atomiclabsnz/datafusion-ducklake#17).
//!
//! Every lookup or comparison of such a name, on every catalog backend, is spelled
//! [`name_eq!`]`(dialect, left, right)`, which expands to a SQL literal; nothing stored
//! changes. A dialect's fold is ASCII-only, as DuckDB's is, so none of SQL's own
//! case-insensitive tools is used where it folds more: DuckDB's and Postgres's `lower()` and
//! DuckDB's `NOCASE` fold `Ä`, and MySQL's default collation is accent-insensitive and pads.
//!
//! - `sqlite`: `COLLATE NOCASE`, which folds ASCII only (an ICU build replaces `lower()`, not
//!   `NOCASE`).
//! - `duckdb`, `postgres`: `translate()` of the 26 ASCII capitals.
//! - `mysql`: the 26 capitals `REPLACE`d one by one over the UTF-8 bytes, compared byte-wise,
//!   since MySQL has no `translate()` and `LOWER()` folds Unicode and ignores a binary string.
//!
//! A comparison in Rust is [`names_match`].

/// Whether two catalog names name the same object: equal but for ASCII case.
pub(crate) fn names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

/// `left = right` under DuckDB's name rule, as a SQL literal for `dialect` (`sqlite`, `duckdb`,
/// `postgres` or `mysql`). `left` and `right` are SQL literals: a column, or a placeholder
/// (`"?"`, `"?1"`, `"$2"`).
macro_rules! name_eq {
    (sqlite, $left:literal, $right:literal) => {
        concat!($left, " = ", $right, " COLLATE NOCASE")
    };
    (duckdb, $left:literal, $right:literal) => {
        $crate::catalog_name::name_eq!(translate, $left, $right)
    };
    (postgres, $left:literal, $right:literal) => {
        $crate::catalog_name::name_eq!(translate, $left, $right)
    };
    (translate, $left:literal, $right:literal) => {
        concat!(
            "translate(",
            $left,
            ", 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = translate(",
            $right,
            ", 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')"
        )
    };
    (mysql, $left:literal, $right:literal) => {
        concat!(
            $crate::catalog_name::name_eq!(mysql_fold, $left),
            " = ",
            $crate::catalog_name::name_eq!(mysql_fold, $right)
        )
    };
    (mysql_fold, $value:literal) => {
        concat!(
            "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(\
             REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(\
             REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(CAST(CONVERT(",
            $value,
            " USING utf8mb4) AS BINARY), \
             'A', 'a'), 'B', 'b'), 'C', 'c'), 'D', 'd'), 'E', 'e'), 'F', 'f'), 'G', 'g'), \
             'H', 'h'), 'I', 'i'), 'J', 'j'), 'K', 'k'), 'L', 'l'), 'M', 'm'), 'N', 'n'), \
             'O', 'o'), 'P', 'p'), 'Q', 'q'), 'R', 'r'), 'S', 's'), 'T', 't'), 'U', 'u'), \
             'V', 'v'), 'W', 'w'), 'X', 'x'), 'Y', 'y'), 'Z', 'z')"
        )
    };
}
pub(crate) use name_eq;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_folds_ascii_only() {
        assert!(names_match("Orders", "orders"));
        assert!(names_match("STAGING", "Staging"));
        assert!(!names_match("Ärger", "ärger"));
        assert!(!names_match("orders", "order"));
    }

    #[test]
    fn every_dialect_folds_the_26_capitals_once() {
        for sql in [
            name_eq!(sqlite, "a", "?"),
            name_eq!(duckdb, "a", "?"),
            name_eq!(postgres, "a", "$1"),
            name_eq!(mysql, "a", "?"),
        ] {
            assert!(sql.contains(" = "), "{sql}");
        }
        let mysql = name_eq!(mysql, "a", "?");
        for (upper, lower) in ('A'..='Z').zip('a'..='z') {
            assert_eq!(
                mysql.matches(&format!("'{upper}', '{lower}')")).count(),
                2,
                "{upper}"
            );
        }
        assert_eq!(mysql.matches("REPLACE(").count(), 52);
    }

    #[cfg(feature = "metadata-duckdb")]
    #[test]
    fn duckdb_and_postgres_fold_is_duckdbs_rule() {
        let connection = duckdb::Connection::open_in_memory().unwrap();
        let sql = concat!("SELECT ", name_eq!(duckdb, "?", "?"));
        for (left, right, same) in [
            ("Orders", "orders", true),
            ("STAGING", "staging", true),
            ("Ärger", "ärger", false),
            ("orders", "order", false),
        ] {
            let answer: bool = connection
                .query_row(sql, [left, right], |row| row.get(0))
                .unwrap();
            assert_eq!(answer, same, "{left} {right}");
            assert_eq!(names_match(left, right), same, "{left} {right}");
        }
    }

    #[cfg(feature = "metadata-sqlite")]
    #[tokio::test]
    async fn sqlite_fold_is_duckdbs_rule() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let sql = concat!("SELECT ", name_eq!(sqlite, "?", "?"));
        for (left, right, same) in [
            ("Orders", "orders", true),
            ("STAGING", "staging", true),
            ("Ärger", "ärger", false),
            ("orders", "order", false),
        ] {
            let answer: bool = sqlx::query_scalar(sql)
                .bind(left)
                .bind(right)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(answer, same, "{left} {right}");
        }
    }
}
