//! The part of sqlx's API the SQLite backend uses, over a [`HostSqlite`].
//!
//! Names and signatures follow sqlx 0.9's, so the backend's source compiles against this module
//! unchanged. Every operation runs when it is called, and the future it returns is already
//! complete. Semantics follow sqlx's SQLite driver where the backend can tell the difference:
//!
//! - a value decodes only as a Rust type of its SQLite storage class (`INTEGER` as an integer
//!   or a `bool`, `REAL` (or `INTEGER`) as an `f64`, `TEXT` as a string, `BLOB` as bytes), and
//!   `NULL` only into an `Option`;
//! - `begin` is a `BEGIN`, a nested `begin` a `SAVEPOINT`, and a transaction dropped without
//!   `commit` rolls back.

use super::{HostError, HostRows, HostSqlite, SQLITE_BUSY, SQLITE_LOCKED, Value};
use std::cell::{Cell, RefCell};
use std::future::{Ready, ready};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

pub mod sqlite {
    pub use super::{
        Sqlite, SqliteConnectOptions, SqliteConnection, SqlitePool, SqlitePoolOptions,
        SqliteQueryResult, SqliteRow,
    };
}

pub mod types {
    pub use chrono;
}

/// The database marker sqlx's generics name.
#[derive(Debug, Clone, Copy)]
pub struct Sqlite;

/// A failed catalog statement.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("error returned from database: {message} (code: {code})")]
    Database {
        code: i32,
        message: String,
    },
    #[error("no rows returned by a query that expected to return at least one row")]
    RowNotFound,
    #[error("no column found for name: {0}")]
    ColumnNotFound(String),
    #[error("column index out of bounds: the len is {len}, but the index is {index}")]
    ColumnIndexOutOfBounds {
        index: usize,
        len: usize,
    },
    #[error("error occurred while decoding column {index}: {message}")]
    ColumnDecode {
        index: String,
        message: String,
    },
    #[error("error with configuration: {0}")]
    Configuration(String),
}

impl Error {
    /// Whether SQLite refused the statement because another connection holds the database.
    pub fn is_busy(&self) -> bool {
        matches!(self, Error::Database { code, .. }
            if code & 0xff == SQLITE_BUSY || code & 0xff == SQLITE_LOCKED)
    }
}

impl From<HostError> for Error {
    fn from(error: HostError) -> Self {
        Error::Database {
            code: error.code,
            message: error.message,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// SQL text

/// sqlx 0.9's marker for SQL built at runtime.
#[derive(Debug, Clone)]
pub struct AssertSqlSafe<T>(pub T);

/// SQL a query may be built from: a literal, or text asserted safe.
pub trait SqlSafeStr {
    fn into_sql(self) -> String;
}

impl SqlSafeStr for &'static str {
    fn into_sql(self) -> String {
        self.to_string()
    }
}

impl SqlSafeStr for AssertSqlSafe<String> {
    fn into_sql(self) -> String {
        self.0
    }
}

impl SqlSafeStr for AssertSqlSafe<&str> {
    fn into_sql(self) -> String {
        self.0.to_string()
    }
}

impl SqlSafeStr for AssertSqlSafe<&String> {
    fn into_sql(self) -> String {
        self.0.clone()
    }
}

// ---------------------------------------------------------------------------------------------
// Values in and out

/// A Rust value bindable to a `?` placeholder.
pub trait Encode {
    fn encode(&self) -> Value;
}

macro_rules! encode_integer {
    ($($ty:ty),*) => {$(
        impl Encode for $ty {
            fn encode(&self) -> Value {
                Value::Integer(i64::from(*self))
            }
        }
    )*};
}
encode_integer!(i8, i16, i32, i64, u8, u16, u32, bool);

impl Encode for f64 {
    fn encode(&self) -> Value {
        Value::Real(*self)
    }
}

impl Encode for f32 {
    fn encode(&self) -> Value {
        Value::Real(f64::from(*self))
    }
}

impl Encode for str {
    fn encode(&self) -> Value {
        Value::Text(self.to_string())
    }
}

impl Encode for String {
    fn encode(&self) -> Value {
        Value::Text(self.clone())
    }
}

impl Encode for [u8] {
    fn encode(&self) -> Value {
        Value::Blob(self.to_vec())
    }
}

impl Encode for Vec<u8> {
    fn encode(&self) -> Value {
        Value::Blob(self.clone())
    }
}

impl Encode for chrono::NaiveDateTime {
    /// As sqlx's SQLite driver writes it.
    fn encode(&self) -> Value {
        Value::Text(self.format("%F %T%.f").to_string())
    }
}

impl<Tz: chrono::TimeZone> Encode for chrono::DateTime<Tz>
where
    Tz::Offset: std::fmt::Display,
{
    /// As sqlx's SQLite driver writes it.
    fn encode(&self) -> Value {
        Value::Text(self.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, false))
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode(&self) -> Value {
        self.as_ref().map_or(Value::Null, Encode::encode)
    }
}

impl<T: Encode + ?Sized> Encode for &T {
    fn encode(&self) -> Value {
        (**self).encode()
    }
}

/// A Rust value a column decodes into.
pub trait Decode: Sized {
    fn decode(value: &Value) -> std::result::Result<Self, String>;
}

fn mismatch(rust: &str, value: &Value) -> String {
    let class = match value {
        Value::Null => "NULL",
        Value::Integer(_) => "INTEGER",
        Value::Real(_) => "REAL",
        Value::Text(_) => "TEXT",
        Value::Blob(_) => "BLOB",
    };
    if matches!(value, Value::Null) {
        format!("unexpected null; try decoding as an `Option` (Rust type `{rust}`)")
    } else {
        format!("mismatched types; Rust type `{rust}` is not compatible with SQL type `{class}`")
    }
}

impl Decode for i64 {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Integer(v) => Ok(*v),
            other => Err(mismatch("i64", other)),
        }
    }
}

macro_rules! decode_narrow_integer {
    ($($ty:ty),*) => {$(
        impl Decode for $ty {
            fn decode(value: &Value) -> std::result::Result<Self, String> {
                let wide = i64::decode(value).map_err(|_| mismatch(stringify!($ty), value))?;
                <$ty>::try_from(wide).map_err(|e| e.to_string())
            }
        }
    )*};
}
decode_narrow_integer!(i8, i16, i32, u8, u16, u32, u64);

impl Decode for bool {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Integer(v) => Ok(*v != 0),
            other => Err(mismatch("bool", other)),
        }
    }
}

impl Decode for f64 {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Real(v) => Ok(*v),
            Value::Integer(v) => Ok(*v as f64),
            other => Err(mismatch("f64", other)),
        }
    }
}

impl Decode for f32 {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        f64::decode(value).map(|v| v as f32)
    }
}

impl Decode for String {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Text(v) => Ok(v.clone()),
            other => Err(mismatch("String", other)),
        }
    }
}

impl Decode for Vec<u8> {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Blob(v) => Ok(v.clone()),
            Value::Text(v) => Ok(v.as_bytes().to_vec()),
            other => Err(mismatch("Vec<u8>", other)),
        }
    }
}

impl Decode for chrono::NaiveDateTime {
    /// The text forms sqlx's SQLite driver reads, and Unix seconds.
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Text(text) => {
                for format in ["%F %T%.f", "%F %T", "%FT%T%.f", "%FT%T", "%F %H:%M", "%FT%H:%M"] {
                    if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(text, format) {
                        return Ok(parsed);
                    }
                }
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|dt| dt.naive_utc())
                    .map_err(|e| format!("not a datetime: {text:?}: {e}"))
            },
            Value::Integer(seconds) => chrono::DateTime::from_timestamp(*seconds, 0)
                .map(|dt| dt.naive_utc())
                .ok_or_else(|| format!("timestamp out of range: {seconds}")),
            other => Err(mismatch("NaiveDateTime", other)),
        }
    }
}

impl Decode for chrono::DateTime<chrono::Utc> {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        chrono::NaiveDateTime::decode(value).map(|naive| naive.and_utc())
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Null => Ok(None),
            other => T::decode(other).map(Some),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Rows

/// One result row.
#[derive(Debug, Clone)]
pub struct SqliteRow {
    columns: Arc<[String]>,
    values: Vec<Value>,
}

/// A column named by position or by name.
pub trait ColumnIndex {
    fn position(&self, columns: &[String]) -> std::result::Result<usize, Error>;
    fn describe(&self) -> String;
}

impl ColumnIndex for usize {
    fn position(&self, columns: &[String]) -> std::result::Result<usize, Error> {
        if *self < columns.len() {
            Ok(*self)
        } else {
            Err(Error::ColumnIndexOutOfBounds {
                index: *self,
                len: columns.len(),
            })
        }
    }
    fn describe(&self) -> String {
        self.to_string()
    }
}

impl ColumnIndex for &str {
    fn position(&self, columns: &[String]) -> std::result::Result<usize, Error> {
        columns
            .iter()
            .position(|c| c == self)
            .ok_or_else(|| Error::ColumnNotFound(self.to_string()))
    }
    fn describe(&self) -> String {
        format!("{self:?}")
    }
}

impl ColumnIndex for &String {
    fn position(&self, columns: &[String]) -> std::result::Result<usize, Error> {
        self.as_str().position(columns)
    }
    fn describe(&self) -> String {
        format!("{self:?}")
    }
}

/// sqlx's `Row`, for the methods the backend calls on a row.
pub trait Row {
    fn try_get<T: Decode, I: ColumnIndex>(&self, index: I) -> std::result::Result<T, Error>;

    /// [`Row::try_get`], panicking on failure, as sqlx's `Row::get` does.
    #[track_caller]
    fn get<T: Decode, I: ColumnIndex>(&self, index: I) -> T {
        match self.try_get(index) {
            Ok(value) => value,
            Err(error) => panic!("{error}"),
        }
    }

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Row for SqliteRow {
    fn try_get<T: Decode, I: ColumnIndex>(&self, index: I) -> std::result::Result<T, Error> {
        let position = index.position(&self.columns)?;
        T::decode(&self.values[position]).map_err(|message| Error::ColumnDecode {
            index: index.describe(),
            message,
        })
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Decoding a whole row, for `query_as`.
pub trait FromRow: Sized {
    fn from_row(row: &SqliteRow) -> std::result::Result<Self, Error>;
}

macro_rules! from_row_tuple {
    ($($name:ident: $index:tt),+) => {
        impl<$($name: Decode),+> FromRow for ($($name,)+) {
            fn from_row(row: &SqliteRow) -> std::result::Result<Self, Error> {
                Ok(($(row.try_get::<$name, usize>($index)?,)+))
            }
        }
    };
}
from_row_tuple!(A: 0);
from_row_tuple!(A: 0, B: 1);
from_row_tuple!(A: 0, B: 1, C: 2);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
from_row_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);

/// What a statement did.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqliteQueryResult {
    changes: u64,
}

impl SqliteQueryResult {
    pub fn rows_affected(&self) -> u64 {
        self.changes
    }
}

// ---------------------------------------------------------------------------------------------
// The connection

thread_local! {
    /// Whether this thread is inside a catalog call ([`super::block_on`]).
    static IN_CALL: Cell<bool> = const { Cell::new(false) };
    /// The connections the current call holds, released when it ends.
    static HELD: RefCell<Vec<Arc<Shared>>> = const { RefCell::new(Vec::new()) };
    /// This thread, as a connection's owner.
    static THREAD: u64 = NEXT_THREAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

static NEXT_THREAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Marks this thread as inside a catalog call for as long as it is held. When the outermost
/// call ends, it commits the implicit read transactions it opened and lets go of the
/// connections it held.
pub(crate) struct CatalogCall {
    outermost: bool,
}

impl CatalogCall {
    pub(crate) fn enter() -> Self {
        let outermost = !IN_CALL.with(|inside| inside.replace(true));
        Self {
            outermost,
        }
    }
}

impl Drop for CatalogCall {
    fn drop(&mut self) {
        if !self.outermost {
            return;
        }
        IN_CALL.with(|inside| inside.set(false));
        for shared in HELD.with(|held| std::mem::take(&mut *held.borrow_mut())) {
            shared.end_call();
        }
    }
}

#[derive(Debug, Default)]
struct State {
    /// Explicit transactions open (the first a `BEGIN`, the rest savepoints).
    depth: u32,
    /// Whether an implicit read transaction is open (only while `depth` is 0).
    implicit: bool,
    /// The thread using the connection: from a catalog call's first statement on it to the
    /// call's end, or from a `begin` outside a call to its transaction's end. A sqlx pool would
    /// give two concurrent calls a connection each; this one connection takes them in turn.
    owner: Option<u64>,
    /// Whether the owner is a catalog call (released when the call ends).
    held_by_call: bool,
}

struct Shared {
    host: Arc<dyn HostSqlite>,
    state: Mutex<State>,
    turn: std::sync::Condvar,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostConnection").finish_non_exhaustive()
    }
}

/// Whether a statement may read only (`fetch_*`) or may write (`execute`).
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fetch,
    Execute,
}

impl Shared {
    /// The connection's state, once this thread holds the connection.
    fn claim(self: &Arc<Self>) -> std::sync::MutexGuard<'_, State> {
        let me = THREAD.with(|thread| *thread);
        // A panic while the lock was held leaves the state as it was before the statement that
        // panicked, which is still the connection's state.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.owner.is_some_and(|owner| owner != me) {
            // Only another thread can hold it, so a single-threaded host never waits here.
            state = self
                .turn
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if state.owner.is_none() {
            state.owner = Some(me);
            if IN_CALL.with(Cell::get) {
                state.held_by_call = true;
                HELD.with(|held| held.borrow_mut().push(Arc::clone(self)));
            }
        }
        state
    }

    /// Let go of the connection if nothing holds it: no transaction, no call.
    fn release_if_idle(&self, state: &mut State) {
        if state.depth == 0 && !state.implicit && !state.held_by_call {
            state.owner = None;
            self.turn.notify_all();
        }
    }

    fn raw(&self, sql: &str) -> std::result::Result<HostRows, Error> {
        Ok(self.host.query(sql, &[])?)
    }

    fn run(
        self: &Arc<Self>,
        sql: &str,
        params: &[Value],
        kind: Kind,
    ) -> std::result::Result<HostRows, Error> {
        let mut state = self.claim();
        let result = self.run_claimed(&mut state, sql, params, kind);
        self.release_if_idle(&mut state);
        result
    }

    fn run_claimed(
        &self,
        state: &mut State,
        sql: &str,
        params: &[Value],
        kind: Kind,
    ) -> std::result::Result<HostRows, Error> {
        if state.depth == 0 {
            match kind {
                // A write outside a transaction runs on its own, as it would on a sqlx pool:
                // the reads before it are over.
                Kind::Execute if state.implicit => {
                    state.implicit = false;
                    self.raw("COMMIT")?;
                },
                Kind::Fetch if !state.implicit && state.held_by_call => {
                    self.raw("BEGIN")?;
                    state.implicit = true;
                },
                _ => {},
            }
        }
        Ok(self.host.query(sql, params)?)
    }

    fn begin(self: &Arc<Self>, statement: Option<&str>) -> std::result::Result<u32, Error> {
        let mut state = self.claim();
        let result = (|| {
            if state.depth == 0 {
                if state.implicit {
                    state.implicit = false;
                    self.raw("COMMIT")?;
                }
                self.raw(statement.unwrap_or("BEGIN"))?;
            } else {
                self.raw(&format!("SAVEPOINT _sqlx_savepoint_{}", state.depth))?;
            }
            state.depth += 1;
            Ok(state.depth)
        })();
        self.release_if_idle(&mut state);
        result
    }

    fn finish(self: &Arc<Self>, depth: u32, commit: bool) -> std::result::Result<(), Error> {
        let mut state = self.claim();
        // A transaction finishes at the depth it began at: its connection is borrowed by the
        // transaction, so an inner one cannot outlive an outer one.
        state.depth = depth - 1;
        let savepoint = format!("_sqlx_savepoint_{}", depth - 1);
        let result = match (depth, commit) {
            // A `COMMIT` SQLite refused (`SQLITE_BUSY`) leaves the transaction open; it is
            // rolled back, so the refusal is the whole outcome and the call can be retried.
            (1, true) => self.raw("COMMIT").map(drop).inspect_err(|_| {
                let _ = self.raw("ROLLBACK");
            }),
            (1, false) => self.raw("ROLLBACK").map(drop),
            (_, true) => self
                .raw(&format!("RELEASE SAVEPOINT {savepoint}"))
                .map(drop),
            (_, false) => self
                .raw(&format!("ROLLBACK TO SAVEPOINT {savepoint}"))
                .and_then(|_| self.raw(&format!("RELEASE SAVEPOINT {savepoint}")))
                .map(drop),
        };
        self.release_if_idle(&mut state);
        result
    }

    /// The end of a catalog call that held this connection: its implicit read transaction is
    /// committed (it read only, so a failed `COMMIT` loses nothing and is rolled back), and
    /// the connection is free for the next call.
    fn end_call(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.depth == 0 && state.implicit {
            state.implicit = false;
            if self.raw("COMMIT").is_err() {
                let _ = self.raw("ROLLBACK");
            }
        }
        state.held_by_call = false;
        self.release_if_idle(&mut state);
    }
}

/// A connection to the host's SQLite. Every clone of a pool shares the one connection.
#[derive(Debug, Clone)]
pub struct SqliteConnection {
    shared: Arc<Shared>,
}

/// sqlx's pool, over the host's one connection.
#[derive(Debug, Clone)]
pub struct SqlitePool {
    connection: SqliteConnection,
}

impl SqlitePool {
    pub(crate) fn from_host(host: Arc<dyn HostSqlite>) -> Self {
        Self {
            connection: SqliteConnection {
                shared: Arc::new(Shared {
                    host,
                    state: Mutex::new(State::default()),
                    turn: std::sync::Condvar::new(),
                }),
            },
        }
    }

    /// As [`SqlitePoolOptions::connect`].
    pub async fn connect(url: &str) -> std::result::Result<Self, Error> {
        connect(url)
    }

    pub fn begin(&self) -> Ready<std::result::Result<Transaction<'static, Sqlite>, Error>> {
        ready(Transaction::begin(self.connection.clone(), None))
    }

    pub fn begin_with(
        &self,
        statement: &'static str,
    ) -> Ready<std::result::Result<Transaction<'static, Sqlite>, Error>> {
        ready(Transaction::begin(self.connection.clone(), Some(statement)))
    }

    pub fn close(&self) -> Ready<()> {
        ready(())
    }
}

/// A transaction: `BEGIN` (or a savepoint inside one) until `commit`; rolled back if dropped.
#[derive(Debug)]
pub struct Transaction<'c, DB> {
    connection: SqliteConnection,
    depth: u32,
    open: bool,
    _marker: PhantomData<(&'c (), DB)>,
}

impl<DB> Transaction<'_, DB> {
    fn begin(
        connection: SqliteConnection,
        statement: Option<&str>,
    ) -> std::result::Result<Self, Error> {
        let depth = connection.shared.begin(statement)?;
        Ok(Self {
            connection,
            depth,
            open: true,
            _marker: PhantomData,
        })
    }

    pub fn commit(mut self) -> Ready<std::result::Result<(), Error>> {
        self.open = false;
        ready(self.connection.shared.finish(self.depth, true))
    }

    pub fn rollback(mut self) -> Ready<std::result::Result<(), Error>> {
        self.open = false;
        ready(self.connection.shared.finish(self.depth, false))
    }
}

impl<DB> Drop for Transaction<'_, DB> {
    fn drop(&mut self) {
        if self.open {
            // As sqlx does on drop: the rollback's own failure has nowhere to go.
            let _ = self.connection.shared.finish(self.depth, false);
        }
    }
}

impl<DB> std::ops::Deref for Transaction<'_, DB> {
    type Target = SqliteConnection;
    fn deref(&self) -> &SqliteConnection {
        &self.connection
    }
}

impl<DB> std::ops::DerefMut for Transaction<'_, DB> {
    fn deref_mut(&mut self) -> &mut SqliteConnection {
        &mut self.connection
    }
}

/// Connection strings are a native concern: a host catalog is built with `from_host`.
#[derive(Debug, Clone, Default)]
pub struct SqlitePoolOptions;

impl SqlitePoolOptions {
    pub fn new() -> Self {
        Self
    }

    pub fn max_connections(self, _max: u32) -> Self {
        self
    }

    pub async fn connect(self, url: &str) -> std::result::Result<SqlitePool, Error> {
        connect(url)
    }

    pub async fn connect_with(
        self,
        options: SqliteConnectOptions,
    ) -> std::result::Result<SqlitePool, Error> {
        connect(&options.url)
    }
}

#[cfg(not(all(test, feature = "metadata-sqlite")))]
fn connect(url: &str) -> std::result::Result<SqlitePool, Error> {
    Err(Error::Configuration(format!(
        "a host SQLite catalog is opened with `from_host`, not a connection string ({url})"
    )))
}

/// In this crate's own tests, a connection string opens a sqlx connection behind the host
/// interface, so the backend's tests run over this module too.
#[cfg(all(test, feature = "metadata-sqlite"))]
fn connect(url: &str) -> std::result::Result<SqlitePool, Error> {
    Ok(SqlitePool::from_host(super::SqlxHost::open(url)?))
}

/// A connection string, kept for the path a native lock file would take.
#[derive(Debug, Clone)]
pub struct SqliteConnectOptions {
    url: String,
}

impl std::str::FromStr for SqliteConnectOptions {
    type Err = Error;
    fn from_str(url: &str) -> std::result::Result<Self, Error> {
        Ok(Self {
            url: url.to_string(),
        })
    }
}

impl SqliteConnectOptions {
    /// No file this crate could lock: a host catalog's commits are serialised by its host.
    pub fn get_filename(&self) -> &std::path::Path {
        std::path::Path::new("")
    }
}

// ---------------------------------------------------------------------------------------------
// Executors and queries

/// Where a query runs: a pool or a connection (a transaction derefs to one).
pub trait Executor<'e> {
    type Database;
    fn run(self, sql: &str, params: &[Value], kind: Kind) -> std::result::Result<HostRows, Error>;
}

impl<'e> Executor<'e> for &'e SqlitePool {
    type Database = Sqlite;
    fn run(self, sql: &str, params: &[Value], kind: Kind) -> std::result::Result<HostRows, Error> {
        self.connection.shared.run(sql, params, kind)
    }
}

impl<'e> Executor<'e> for &'e mut SqliteConnection {
    type Database = Sqlite;
    fn run(self, sql: &str, params: &[Value], kind: Kind) -> std::result::Result<HostRows, Error> {
        self.shared.run(sql, params, kind)
    }
}

fn rows_of(result: HostRows) -> Vec<SqliteRow> {
    let columns: Arc<[String]> = result.columns.into();
    result
        .rows
        .into_iter()
        .map(|values| SqliteRow {
            columns: Arc::clone(&columns),
            values,
        })
        .collect()
}

/// `sqlx::query`.
pub struct Query<DB = Sqlite> {
    sql: String,
    params: Vec<Value>,
    _marker: PhantomData<DB>,
}

pub fn query<DB>(sql: impl SqlSafeStr) -> Query<DB> {
    Query {
        sql: sql.into_sql(),
        params: Vec::new(),
        _marker: PhantomData,
    }
}

impl<DB> Query<DB> {
    pub fn bind<T: Encode>(mut self, value: T) -> Self {
        self.params.push(value.encode());
        self
    }

    /// Statement caching is the host's; accepted and ignored.
    pub fn persistent(self, _persistent: bool) -> Self {
        self
    }

    pub fn execute<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<SqliteQueryResult, Error>> {
        ready(
            executor
                .run(&self.sql, &self.params, Kind::Execute)
                .map(|result| SqliteQueryResult {
                    changes: result.changes,
                }),
        )
    }

    pub fn fetch_all<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Vec<SqliteRow>, Error>> {
        ready(
            executor
                .run(&self.sql, &self.params, Kind::Fetch)
                .map(rows_of),
        )
    }

    pub fn fetch_optional<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Option<SqliteRow>, Error>> {
        ready(
            executor
                .run(&self.sql, &self.params, Kind::Fetch)
                .map(|result| rows_of(result).into_iter().next()),
        )
    }

    pub fn fetch_one<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<SqliteRow, Error>> {
        ready(match self.fetch_optional(executor).into_inner() {
            Ok(Some(row)) => Ok(row),
            Ok(None) => Err(Error::RowNotFound),
            Err(error) => Err(error),
        })
    }
}

/// `sqlx::query_scalar`: the first column of each row.
pub struct QueryScalar<DB, O> {
    inner: Query<DB>,
    _marker: PhantomData<O>,
}

pub fn query_scalar<DB, O>(sql: impl SqlSafeStr) -> QueryScalar<DB, O> {
    QueryScalar {
        inner: query(sql),
        _marker: PhantomData,
    }
}

fn first_column<O: Decode>(row: &SqliteRow) -> std::result::Result<O, Error> {
    row.try_get(0usize)
}

impl<DB, O: Decode> QueryScalar<DB, O> {
    pub fn bind<T: Encode>(mut self, value: T) -> Self {
        self.inner = self.inner.bind(value);
        self
    }

    pub fn fetch_all<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Vec<O>, Error>> {
        ready(
            self.inner
                .fetch_all(executor)
                .into_inner()
                .and_then(|rows| rows.iter().map(first_column).collect()),
        )
    }

    pub fn fetch_optional<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Option<O>, Error>> {
        ready(
            self.inner
                .fetch_optional(executor)
                .into_inner()
                .and_then(|row| row.as_ref().map(first_column).transpose()),
        )
    }

    pub fn fetch_one<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<O, Error>> {
        ready(
            self.inner
                .fetch_one(executor)
                .into_inner()
                .and_then(|row| first_column(&row)),
        )
    }
}

/// `sqlx::query_as`: each row decoded whole.
pub struct QueryAs<DB, O> {
    inner: Query<DB>,
    _marker: PhantomData<O>,
}

pub fn query_as<DB, O>(sql: impl SqlSafeStr) -> QueryAs<DB, O> {
    QueryAs {
        inner: query(sql),
        _marker: PhantomData,
    }
}

impl<DB, O: FromRow> QueryAs<DB, O> {
    pub fn bind<T: Encode>(mut self, value: T) -> Self {
        self.inner = self.inner.bind(value);
        self
    }

    pub fn fetch_all<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Vec<O>, Error>> {
        ready(
            self.inner
                .fetch_all(executor)
                .into_inner()
                .and_then(|rows| rows.iter().map(O::from_row).collect()),
        )
    }

    pub fn fetch_optional<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<Option<O>, Error>> {
        ready(
            self.inner
                .fetch_optional(executor)
                .into_inner()
                .and_then(|row| row.as_ref().map(O::from_row).transpose()),
        )
    }

    pub fn fetch_one<'e, E: Executor<'e, Database = DB>>(
        self,
        executor: E,
    ) -> Ready<std::result::Result<O, Error>> {
        ready(
            self.inner
                .fetch_one(executor)
                .into_inner()
                .and_then(|row| O::from_row(&row)),
        )
    }
}

/// `sqlx::QueryBuilder`: SQL text with `?` placeholders pushed beside their values.
pub struct QueryBuilder<DB> {
    sql: String,
    params: Vec<Value>,
    _marker: PhantomData<DB>,
}

impl<DB> QueryBuilder<DB> {
    pub fn new(init: impl Into<String>) -> Self {
        Self {
            sql: init.into(),
            params: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push(&mut self, sql: impl std::fmt::Display) -> &mut Self {
        use std::fmt::Write as _;
        // Writing to a `String` cannot fail.
        let _ = write!(self.sql, "{sql}");
        self
    }

    pub fn push_bind<T: Encode>(&mut self, value: T) -> &mut Self {
        self.sql.push('?');
        self.params.push(value.encode());
        self
    }

    /// `VALUES (…), (…)`, one tuple per item, as sqlx writes it.
    pub fn push_values<I, F>(&mut self, tuples: I, mut push_tuple: F) -> &mut Self
    where
        I: IntoIterator,
        F: FnMut(Separated<'_, DB>, I::Item),
    {
        self.sql.push_str("VALUES ");
        for (i, item) in tuples.into_iter().enumerate() {
            if i > 0 {
                self.sql.push_str(", ");
            }
            self.sql.push('(');
            push_tuple(
                Separated {
                    builder: self,
                    first: true,
                },
                item,
            );
            self.sql.push(')');
        }
        self
    }

    pub fn build(&mut self) -> Query<DB> {
        Query {
            sql: std::mem::take(&mut self.sql),
            params: std::mem::take(&mut self.params),
            _marker: PhantomData,
        }
    }
}

/// One `VALUES` tuple being pushed, its items separated by commas.
pub struct Separated<'b, DB> {
    builder: &'b mut QueryBuilder<DB>,
    first: bool,
}

impl<DB> Separated<'_, DB> {
    pub fn push_bind<T: Encode>(&mut self, value: T) -> &mut Self {
        if !self.first {
            self.builder.sql.push_str(", ");
        }
        self.first = false;
        self.builder.push_bind(value);
        self
    }
}
