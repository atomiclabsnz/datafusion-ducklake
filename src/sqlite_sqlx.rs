//! The SQLite backend over sqlx.
//!
//! [`metadata_provider_sqlite`] and [`metadata_writer_sqlite`] name sqlx, the runtime bridge and
//! each other through `super::`, so the same two files also compile over a host's own SQLite
//! ([`crate::host_sqlite`]). Here `super::` is native: sqlx itself and the catalog runtime.

pub(crate) use crate::metadata_provider::{block_on, connect_on_catalog_runtime};
pub(crate) use ::sqlx;

#[path = "metadata_provider_sqlite.rs"]
pub mod metadata_provider_sqlite;
#[cfg(feature = "write-sqlite")]
#[path = "metadata_writer_sqlite.rs"]
pub mod metadata_writer_sqlite;

/// The native writer's maintenance: expiry and file cleanup ([`crate::maintenance`]).
#[cfg(feature = "write-sqlite")]
pub(crate) fn maintenance(
    writer: &metadata_writer_sqlite::SqliteMetadataWriter,
) -> Option<&dyn crate::maintenance::CatalogMaintenance> {
    Some(writer)
}
