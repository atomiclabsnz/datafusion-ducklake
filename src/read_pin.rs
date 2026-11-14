//! [`ReadPin`]: one head read for every lookup and scan of a read-only statement, and the
//! tables built at a head kept until the head moves.
//!
//! A catalog from [`DuckLakeCatalog::new`](crate::DuckLakeCatalog::new) or
//! [`DuckLakeCatalog::with_writer`](crate::DuckLakeCatalog::with_writer) reads the catalog on
//! every lookup: the head, the schema, the head again, the table, its columns, its partition
//! spec and its statistics. Each scan then reads the listing, the inlined deletions and the
//! inlined rows. DataFusion looks each table up at least once per statement, so a statement
//! over three tables makes some thirty catalog calls before it reads a row. Over a SQLite
//! that takes a file lock per call (a browser's OPFS SQLite, about a millisecond each), that
//! is most of a small query's time (ferrion#846).
//!
//! [`DuckLakeCatalog::with_read_pins`](crate::DuckLakeCatalog::with_read_pins) turns on this
//! opt-in path. While a [`ReadPin`] from
//! [`DuckLakeCatalog::pin_reads`](crate::DuckLakeCatalog::pin_reads) is held:
//!
//! - every lookup reads the head the pin read, so the statement makes **one** head read;
//! - a schema and a table found at that head are kept, keyed by the head, and served to the
//!   next lookup at the same head without reading the catalog;
//! - each table served goes through the pin's provider, bound to a [`CatalogReadMemo`], so its
//!   scans' listing, inlined deletions and inlined rows are read once per head as well; that
//!   provider answers "the current snapshot" with the pin's head, which is what it is for the
//!   statement.
//!
//! # Correctness
//!
//! It rests on the facts [`CatalogReadMemo`] documents. The one that matters here is that every
//! commit that changes what a lookup or a scan reads moves the head: a new table, a dropped
//! one, a column added, a file written or deleted. A pin that reads a newer head drops every
//! schema and table kept, as the memo drops its fills. The memo's other caveats (expired
//! snapshots, the low-level [`MetadataWriter`](crate::metadata_writer::MetadataWriter) calls
//! after `create_snapshot`, a catalog restored to an older head) hold here too; a head that is
//! *older* than the kept one also drops what is kept.
//!
//! # Reads only
//!
//! A pin is for a statement that reads. A table served under a pin has no writer, so an
//! `INSERT`, `DELETE` or `UPDATE` planned under one is refused as on a read-only catalog rather
//! than written against kept reads. Take the pin after deciding the statement reads, and drop
//! it before planning one that writes. Schemas served under a pin keep the catalog's writer,
//! so DDL still commits through it (a commit reads the catalog itself, in its own transaction).
//!
//! Without a pin the catalog reads as it always does: on every lookup.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::Result;
use crate::catalog_read_memo::CatalogReadMemo;
use crate::metadata_provider::{MetadataProvider, SchemaMetadata};
use crate::table::DuckLakeTable;

/// What a catalog keeps for its pins: the memo its pins bind, and the lookups found at the
/// newest head a pin read.
pub(crate) struct ReadPins {
    memo: Arc<CatalogReadMemo>,
    state: Mutex<PinState>,
}

#[derive(Default)]
struct PinState {
    /// Pins held. Lookups use the pin while this is above zero.
    held: usize,
    /// The pin's head and its provider, bound to the memo, while one is held.
    pinned: Option<Pinned>,
    /// The head what is kept below was found at.
    kept_head: Option<i64>,
    /// Schemas by the name they were looked up by; `None` where the name found none.
    schemas: HashMap<String, Option<SchemaMetadata>>,
    /// Tables by schema id and the name they were looked up by, built without a writer.
    tables: HashMap<(i64, String), DuckLakeTable>,
}

/// The head a held pin read, and the provider its lookups' tables read through.
#[derive(Clone)]
pub(crate) struct Pinned {
    pub(crate) head: i64,
    pub(crate) provider: Arc<dyn MetadataProvider>,
}

impl ReadPins {
    pub(crate) fn new(memo_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            memo: CatalogReadMemo::new(memo_bytes),
            state: Mutex::default(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, PinState> {
        // A panic while the lock was held leaves maps whose entries were each complete.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Pin `provider`'s head: read it once (the memo's bind), and drop what is kept when the
    /// head is not the one it was kept at. A pin taken while one is held shares it.
    pub(crate) fn pin(self: &Arc<Self>, provider: &Arc<dyn MetadataProvider>) -> Result<ReadPin> {
        {
            let mut state = self.lock();
            if state.held > 0 {
                state.held += 1;
                return Ok(ReadPin {
                    pins: Arc::clone(self),
                });
            }
        }
        let bound = self.memo.bind_pinned(Arc::clone(provider))?;
        let head = bound.head();
        let mut state = self.lock();
        state.held += 1;
        // Another pin can have been taken while this one read the head (another thread): the
        // first one's head stands for both, as it would have for a pin taken after it, and
        // what is kept is left at its head.
        if state.pinned.is_none() {
            if state.kept_head != Some(head) {
                state.kept_head = Some(head);
                state.schemas.clear();
                state.tables.clear();
            }
            state.pinned = Some(Pinned {
                head,
                provider: Arc::new(bound),
            });
        }
        Ok(ReadPin {
            pins: Arc::clone(self),
        })
    }

    /// The held pin, if there is one.
    pub(crate) fn pinned(&self) -> Option<Pinned> {
        self.lock().pinned.clone()
    }

    /// The schema named `name` at the pin's head, from what is kept or else by `read`, which
    /// is then kept.
    pub(crate) fn schema(
        &self,
        pinned: &Pinned,
        name: &str,
        read: impl FnOnce() -> Result<Option<SchemaMetadata>>,
    ) -> Result<Option<SchemaMetadata>> {
        if let Some(kept) = self.kept(pinned, |state| state.schemas.get(name).cloned()) {
            return Ok(kept);
        }
        let found = read()?;
        self.keep(pinned, |state| {
            state.schemas.insert(name.to_string(), found.clone());
        });
        Ok(found)
    }

    /// The table `name` of schema `schema_id` kept at the pin's head.
    pub(crate) fn table(
        &self,
        pinned: &Pinned,
        schema_id: i64,
        name: &str,
    ) -> Option<DuckLakeTable> {
        self.kept(pinned, |state| {
            state.tables.get(&(schema_id, name.to_string())).cloned()
        })
    }

    /// Keep `table`, found as `name` in schema `schema_id` at the pin's head.
    pub(crate) fn keep_table(
        &self,
        pinned: &Pinned,
        schema_id: i64,
        name: &str,
        table: &DuckLakeTable,
    ) {
        self.keep(pinned, |state| {
            state
                .tables
                .insert((schema_id, name.to_string()), table.clone());
        });
    }

    /// What `get` finds among what is kept, only while it was kept at `pinned`'s head.
    fn kept<T>(&self, pinned: &Pinned, get: impl FnOnce(&PinState) -> Option<T>) -> Option<T> {
        let state = self.lock();
        if state.kept_head != Some(pinned.head) {
            return None;
        }
        get(&state)
    }

    /// Keep with `put`, only while what is kept is at `pinned`'s head.
    fn keep(&self, pinned: &Pinned, put: impl FnOnce(&mut PinState)) {
        let mut state = self.lock();
        if state.kept_head == Some(pinned.head) {
            put(&mut state);
        }
    }

    fn unpin(&self) {
        let mut state = self.lock();
        state.held = state.held.saturating_sub(1);
        if state.held == 0 {
            state.pinned = None;
        }
    }
}

impl fmt::Debug for ReadPins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("ReadPins")
            .field("held", &state.held)
            .field("kept_head", &state.kept_head)
            .field("schemas", &state.schemas.len())
            .field("tables", &state.tables.len())
            .field("memo", &self.memo)
            .finish()
    }
}

/// A held pin on a catalog's head, from
/// [`DuckLakeCatalog::pin_reads`](crate::DuckLakeCatalog::pin_reads). Lookups read the head it
/// read until it is dropped; see the [module documentation](self).
///
/// A table taken under a pin keeps its pin's reads after the pin is dropped: a plan built
/// under the pin runs on them. Drop the pin when the statement's plan is built, or after it
/// has run.
#[must_use = "lookups use the pin only while it is held"]
pub struct ReadPin {
    pins: Arc<ReadPins>,
}

impl ReadPin {
    /// The head this pin's lookups read.
    pub fn head(&self) -> Option<i64> {
        self.pins.pinned().map(|pinned| pinned.head)
    }
}

impl Drop for ReadPin {
    fn drop(&mut self) {
        self.pins.unpin();
    }
}

impl fmt::Debug for ReadPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadPin")
            .field("head", &self.head())
            .finish()
    }
}
