//! DuckLake's SQLite catalog over a SQLite the host supplies.
//!
//! The SQLite backend ([`crate::metadata_provider_sqlite`], [`crate::metadata_writer_sqlite`])
//! is written against sqlx. This module compiles **the same two source files** a second time,
//! with `sqlx` resolved to [`sqlx`], a small module of the same shape over a [`HostSqlite`]:
//! one synchronous call that runs SQL and returns its rows. So a host that already has a SQLite
//! (a browser page's `@sqlite.org/sqlite-wasm` `oo1.OpfsDb`, say) can hand it to DuckLake, and
//! the catalog is read and written with exactly the SQL the native backend runs: one source,
//! no copy to drift.
//!
//! Every call into a [`HostSqlite`] returns before it returns its future, so each catalog call
//! completes in one poll and needs no async runtime. It also means no catalog transaction can
//! span an `await` of anything else. Three more rules hold for every catalog call:
//!
//! - **One transaction per call.** A host SQLite on OPFS pays about a millisecond to take its
//!   lock for a statement outside a transaction. Reads a catalog call makes outside an explicit
//!   transaction therefore share one implicit transaction, committed when the call returns.
//! - **`SQLITE_BUSY` is a typed refusal.** It surfaces as
//!   [`crate::DuckLakeError::CatalogBusy`], which says to retry, not as a generic error.
//! - **The connection is the host's.** Its pragmas (`locking_mode`, journal mode) are left as
//!   the host opened it.
//!
//! [`HostSqliteMetadataProvider`] and [`HostSqliteMetadataWriter`] are the native
//! `SqliteMetadataProvider` and `SqliteMetadataWriter`, instantiated over this module; build
//! them with `from_host`. A host creating a catalog from nothing calls
//! `initialize_official_schema`, which creates the catalog the DuckDB extension creates, so the
//! lake stays one DuckDB attaches.

use std::sync::Arc;

pub mod sqlx;

// The same files as the native backend's, on purpose (see the module doc).
#[allow(clippy::duplicate_mod)]
#[path = "metadata_provider_sqlite.rs"]
pub mod metadata_provider_sqlite;
// Also the native writer's file. Its expiry and cleanup helpers serve only the native
// maintenance (`crate::maintenance`), which a host catalog does not have yet; the native
// instantiation keeps them checked for dead code.
#[allow(clippy::duplicate_mod, dead_code)]
#[path = "metadata_writer_sqlite.rs"]
pub mod metadata_writer_sqlite;

#[cfg(feature = "metadata-sqlite")]
mod sqlx_host;
#[cfg(feature = "metadata-sqlite")]
pub use sqlx_host::SqlxHost;

pub use metadata_provider_sqlite::SqliteMetadataProvider as HostSqliteMetadataProvider;
pub use metadata_writer_sqlite::SqliteMetadataWriter as HostSqliteMetadataWriter;
pub use sqlx::Error;

/// One SQLite value, as SQLite stores it.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

/// What one [`HostSqlite::query`] returned.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HostRows {
    /// The result's column names, in order.
    pub columns: Vec<String>,
    /// Each row's values, one per column.
    pub rows: Vec<Vec<Value>>,
    /// Rows the statement inserted, updated or deleted (`sqlite3_changes`).
    pub changes: u64,
}

/// A statement the host's SQLite refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    /// SQLite's result code (`SQLITE_BUSY` is 5); a primary or extended code.
    pub code: i32,
    /// SQLite's message.
    pub message: String,
}

/// `SQLITE_BUSY`.
pub const SQLITE_BUSY: i32 = 5;
/// `SQLITE_LOCKED`.
pub const SQLITE_LOCKED: i32 = 6;

/// A synchronous connection to a SQLite database, supplied by the host.
///
/// `query` runs `sql` with `params` bound to its `?` placeholders in order, and returns its rows
/// before it returns. `sql` holds one statement when `params` is non-empty; without parameters
/// it may hold several (a schema script), and the rows are the last statement's. The host must
/// not run statements of its own on the connection between two calls of one catalog call:
/// transactions DuckLake opens with `BEGIN` are on this connection.
pub trait HostSqlite: Send + Sync {
    fn query(&self, sql: &str, params: &[Value]) -> std::result::Result<HostRows, HostError>;
}

impl std::fmt::Debug for dyn HostSqlite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostSqlite")
    }
}

/// Drive a catalog future to completion. Its awaits are all on [`sqlx`], which completes each
/// before returning it, so one poll is enough; the reads it makes outside a transaction share
/// one, committed here (see the module doc).
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let _call = sqlx::CatalogCall::enter();
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => output,
        // Unreachable while every await in the SQLite backend is on `sqlx` (this module's), each
        // of which is ready when it is created. A future that waited on anything else would
        // have no runtime here to wake it, so failing loudly is the only honest answer.
        std::task::Poll::Pending => panic!(
            "a DuckLake catalog call over a host SQLite awaited something other than the host"
        ),
    }
}

/// The native backend opens its pool on a runtime of its own; a host connection needs none.
pub(crate) async fn connect_on_catalog_runtime<F, T>(connect: F) -> T
where
    F: std::future::Future<Output = T>,
{
    connect.await
}

/// The writer's maintenance: expiry and file cleanup are written for the native writer, so a
/// host catalog has none yet.
pub(crate) fn maintenance(
    _writer: &metadata_writer_sqlite::SqliteMetadataWriter,
) -> Option<&dyn crate::maintenance::CatalogMaintenance> {
    None
}

impl metadata_provider_sqlite::SqliteMetadataProvider {
    /// A provider over the host's SQLite.
    pub fn from_host(host: Arc<dyn HostSqlite>) -> Self {
        Self::from_pool(sqlx::SqlitePool::from_host(host))
    }
}

impl metadata_writer_sqlite::SqliteMetadataWriter {
    /// A writer over the host's SQLite. Like the native writer's `new`, it migrates an existing
    /// catalog; call [`crate::MetadataWriter::initialize_schema`] for a new one.
    pub fn from_host(host: Arc<dyn HostSqlite>) -> crate::Result<Self> {
        block_on(Self::from_pool(sqlx::SqlitePool::from_host(host)))
    }
}
