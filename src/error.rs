//! Error types for the DuckLake DataFusion extension

use std::fmt;

use thiserror::Error;

/// The data-write mode that attempted an unsupported column type change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeChangeWriteMode {
    /// Drop existing data and replace with new data.
    Replace,
    /// Keep existing data and append new records.
    Append,
}

impl fmt::Display for TypeChangeWriteMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeChangeWriteMode::Replace => write!(f, "Replace"),
            TypeChangeWriteMode::Append => write!(f, "Append"),
        }
    }
}

/// The operation that attempted an unsupported column type change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeChangeOperation {
    /// Explicit metadata-only schema evolution through `promote_column_type`.
    PromoteColumnType,
    /// A data write tried to change the type of an existing same-name column.
    DataWrite {
        mode: TypeChangeWriteMode,
    },
}

impl fmt::Display for TypeChangeOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeChangeOperation::PromoteColumnType => write!(f, "promote_column_type"),
            TypeChangeOperation::DataWrite {
                mode,
            } => write!(f, "{mode} data write"),
        }
    }
}

/// Error type for DuckLake operations
#[derive(Error, Debug)]
pub enum DuckLakeError {
    /// Error from DataFusion
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    /// Error from Arrow
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// DuckDB error
    #[cfg(feature = "metadata-duckdb")]
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    /// sqlx database error (for PostgreSQL/MySQL/SQLite metadata providers)
    #[cfg(any(
        feature = "metadata-postgres",
        feature = "metadata-mysql",
        feature = "metadata-sqlite"
    ))]
    #[error("Database error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// A statement a host's SQLite refused ([`crate::host_sqlite`]).
    #[cfg(feature = "host-sqlite")]
    #[error("Host SQLite error: {0}")]
    HostSqlite(crate::host_sqlite::Error),

    /// A host's SQLite answered `SQLITE_BUSY` (or `SQLITE_LOCKED`): another connection holds the
    /// catalog. The refused statement's transaction was rolled back; retry the call.
    #[cfg(feature = "host-sqlite")]
    #[error("catalog busy, retry: {0}")]
    CatalogBusy(String),

    /// The `COMMIT` of a metadata transaction failed on a PostgreSQL or MySQL
    /// catalog, so whether it took effect is unknown: the server may have
    /// committed it before the connection failed. Raised by the commits that
    /// register data files and by `commit_multi_table`; any other error from
    /// those means nothing was committed. Read the catalog before retrying.
    #[cfg(any(feature = "metadata-postgres", feature = "metadata-mysql"))]
    #[error("Commit outcome unknown: {0}")]
    CommitOutcomeUnknown(#[source] sqlx::Error),

    /// A PostgreSQL metadata writer's commit guard refused a commit, so its
    /// transaction was rolled back and nothing was committed. The source is the
    /// error the guard returned; see
    /// [`PostgresCommitGuard`](crate::metadata_writer_postgres::PostgresCommitGuard).
    #[cfg(feature = "write-postgres")]
    #[error("Commit refused: {0}")]
    CommitRefused(#[source] crate::metadata_writer_postgres::CommitGuardError),

    /// Catalog not found
    #[error("Catalog not found: {0}")]
    CatalogNotFound(String),

    /// Schema not found
    #[error("Schema not found: {0}")]
    SchemaNotFound(String),

    /// Table not found
    #[error("Table not found: {0}")]
    TableNotFound(String),

    /// A [`WriteMode::Create`](crate::metadata_writer::WriteMode::Create) write met
    /// a table that exists, or that another writer created before it committed.
    #[error("Table already exists: {0}")]
    TableExists(String),

    /// A write that would make a table met a live view with the table's name, there
    /// before it began or created by another writer before it committed. The view is
    /// left as it is, whatever the write's mode: a table never stands beside a view of
    /// its name.
    #[error("A view has that name: {0}")]
    ViewExists(String),

    /// Invalid snapshot
    #[error("Invalid snapshot: {0}")]
    InvalidSnapshot(String),

    /// Invalid catalog configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A write or promotion tried to change an existing column to a type that
    /// DuckLake cannot adopt through metadata-only schema evolution.
    #[error(
        "Unsupported type change during {operation}: column '{column}' from '{from}' to '{to}'"
    )]
    UnsupportedTypeChange {
        operation: TypeChangeOperation,
        column: String,
        from: String,
        to: String,
    },

    /// Unsupported DuckLake type
    #[error("Unsupported DuckLake type: {0}")]
    UnsupportedType(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// ObjectStore error
    #[error("ObjectStore error: {0}")]
    ObjectStore(#[from] object_store::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parquet error
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// A concurrent write conflict detected at commit time: another writer
    /// published a newer generation of the table since this write began. The
    /// loser aborts (DuckLake-style optimistic concurrency) rather than silently
    /// unioning or clobbering the concurrent commit. Callers may retry.
    #[error("Write conflict: {0}")]
    Conflict(String),

    /// A file manifest, or one of its files, cannot be imported.
    #[cfg(feature = "write")]
    #[error("Manifest import refused: {0}")]
    Import(#[from] crate::manifest_import::ImportError),

    /// A table snapshot cannot be exported as a file manifest.
    #[error("Manifest export refused: {0}")]
    Manifest(#[from] crate::manifest::ManifestError),

    /// Generic error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<DuckLakeError> for datafusion::error::DataFusionError {
    fn from(err: DuckLakeError) -> Self {
        match err {
            // If it's already a DataFusion error, unwrap it
            DuckLakeError::DataFusion(e) => e,
            // For all other errors, wrap them as External
            other => datafusion::error::DataFusionError::External(Box::new(other)),
        }
    }
}

#[cfg(feature = "host-sqlite")]
impl From<crate::host_sqlite::Error> for DuckLakeError {
    fn from(error: crate::host_sqlite::Error) -> Self {
        if error.is_busy() {
            DuckLakeError::CatalogBusy(error.to_string())
        } else {
            DuckLakeError::HostSqlite(error)
        }
    }
}
