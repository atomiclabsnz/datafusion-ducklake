//! A [`HostSqlite`] over a native sqlx connection: the host backend, run natively. This crate's
//! unit tests of the SQLite backend run a second time through it, and its integration tests
//! use it to put DuckDB on the other side of a host catalog.

use super::{HostError, HostRows, HostSqlite, Value};
use ::sqlx::{Column as _, Connection as _, Row as _, TypeInfo as _, ValueRef as _};
use std::sync::Arc;

/// One native SQLite connection behind [`HostSqlite`].
pub struct SqlxHost {
    connection: Arc<tokio::sync::Mutex<::sqlx::SqliteConnection>>,
}

impl SqlxHost {
    /// Open `url` (`sqlite:path?mode=rwc`, `sqlite::memory:`) with sqlx's default busy timeout.
    pub fn open(url: &str) -> std::result::Result<Arc<dyn HostSqlite>, super::Error> {
        Self::open_with_busy_timeout(url, std::time::Duration::from_secs(5))
    }

    /// Open `url`, waiting at most `busy_timeout` for another connection's lock before a
    /// statement fails with `SQLITE_BUSY`.
    pub fn open_with_busy_timeout(
        url: &str,
        busy_timeout: std::time::Duration,
    ) -> std::result::Result<Arc<dyn HostSqlite>, super::Error> {
        let options = <::sqlx::sqlite::SqliteConnectOptions as std::str::FromStr>::from_str(url)
            .map_err(|error| super::Error::Configuration(error.to_string()))?
            .busy_timeout(busy_timeout);
        let connection = crate::metadata_provider::block_on(
            crate::metadata_provider::connect_on_catalog_runtime(async move {
                ::sqlx::SqliteConnection::connect_with(&options).await
            }),
        )
        .map_err(|error| super::Error::Configuration(error.to_string()))?;
        Ok(Arc::new(Self {
            connection: Arc::new(tokio::sync::Mutex::new(connection)),
        }))
    }
}

fn host_error(error: ::sqlx::Error) -> HostError {
    let code = match &error {
        ::sqlx::Error::Database(db) => db
            .code()
            .and_then(|code| code.parse::<i32>().ok())
            .unwrap_or(1),
        _ => 1,
    };
    HostError {
        code,
        message: match &error {
            ::sqlx::Error::Database(db) => db.message().to_string(),
            other => other.to_string(),
        },
    }
}

fn value_of(
    row: &::sqlx::sqlite::SqliteRow,
    index: usize,
) -> std::result::Result<Value, HostError> {
    let raw = row.try_get_raw(index).map_err(host_error)?;
    if raw.is_null() {
        return Ok(Value::Null);
    }
    let class = raw.type_info().name().to_string();
    let value = match class.as_str() {
        "INTEGER" => Value::Integer(row.try_get_unchecked::<i64, _>(index).map_err(host_error)?),
        "REAL" => Value::Real(row.try_get_unchecked::<f64, _>(index).map_err(host_error)?),
        "BLOB" => Value::Blob(
            row.try_get_unchecked::<Vec<u8>, _>(index)
                .map_err(host_error)?,
        ),
        _ => Value::Text(
            row.try_get_unchecked::<String, _>(index)
                .map_err(host_error)?,
        ),
    };
    Ok(value)
}

impl HostSqlite for SqlxHost {
    fn query(&self, sql: &str, params: &[Value]) -> std::result::Result<HostRows, HostError> {
        let connection = Arc::clone(&self.connection);
        let sql = sql.to_string();
        let params = params.to_vec();
        crate::metadata_provider::block_on(async move {
            let mut connection = connection.lock().await;
            let rows = if params.is_empty() {
                ::sqlx::raw_sql(::sqlx::AssertSqlSafe(sql))
                    .fetch_all(&mut *connection)
                    .await
            } else {
                let mut query = ::sqlx::query(::sqlx::AssertSqlSafe(sql));
                for param in params {
                    query = match param {
                        Value::Null => query.bind(None::<i64>),
                        Value::Integer(v) => query.bind(v),
                        Value::Real(v) => query.bind(v),
                        Value::Text(v) => query.bind(v),
                        Value::Blob(v) => query.bind(v),
                    };
                }
                query.fetch_all(&mut *connection).await
            }
            .map_err(host_error)?;
            let changes: i64 = ::sqlx::query_scalar("SELECT changes()")
                .fetch_one(&mut *connection)
                .await
                .map_err(host_error)?;
            let columns = rows
                .first()
                .map(|row| row.columns().iter().map(|c| c.name().to_string()).collect())
                .unwrap_or_default();
            let rows = rows
                .iter()
                .map(|row| (0..row.len()).map(|i| value_of(row, i)).collect())
                .collect::<std::result::Result<Vec<Vec<Value>>, HostError>>()?;
            Ok(HostRows {
                columns,
                rows,
                changes: u64::try_from(changes).unwrap_or(0),
            })
        })
    }
}
