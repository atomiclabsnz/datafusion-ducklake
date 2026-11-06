//! Where a Parquet file waits between the synchronous writer that encodes it and
//! the asynchronous upload that stores it.
//!
//! [`Staging::File`] is a local temporary file: a multi-GB table is never held in
//! RAM, and the upload streams it out through a multipart upload. It is the
//! default wherever there is a filesystem.
//!
//! [`Staging::Memory`] holds the encoded bytes in memory and uploads them with a
//! single `put`. It is available everywhere (`wasm32-unknown-unknown` has no
//! filesystem: `std::fs` compiles there and fails every call). A staged file is
//! bounded by the writer's `target_file_size`, which is how a caller bounds the
//! memory this costs.
//!
//! [`Staging::Parts`] sends a file out as it is encoded: the Parquet writer encodes
//! into a buffer, and once the buffer holds a part, an async caller
//! ([`StageWriter::put_full_parts`], reached through `write_batch_async`) awaits
//! [`MultipartUpload::put_part`] for it inline. The footer goes as the last part,
//! and `complete` makes the object. An open file then holds at most one flushed
//! row group (until its parts are sent), the row group in progress and under one
//! part, however large the file. It is the default on `wasm32`. A file of which
//! no part was sent before its footer (one below a part, one written through the
//! synchronous `write_batch`, one whose only row group closes at its finish) is
//! never opened as an upload and goes out in one `put`, at any size. The file's
//! statistics come from the `ParquetMetaData` the writer returns whenever there is
//! one, since a sent file's bytes are gone by the time it is finished.
//!
//! Neither in-memory staging spawns a task: `object_store`'s own multipart writer
//! (`WriteMultipart`) spawns one per part, and a browser has no tokio runtime to
//! run them on. A part is awaited where it is produced instead.

use std::io::Write;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutPayload};
use parquet::file::metadata::ParquetMetaData;

use crate::Result;
use crate::metadata_writer::ColumnStat;
use crate::stats_collect::WrittenLeafStats;

/// How a writer stages each Parquet file before upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Staging {
    /// A local temporary file, streamed out through a multipart upload.
    /// Not available on `wasm32`. The default elsewhere.
    #[cfg(not(target_arch = "wasm32"))]
    File,
    /// The encoded bytes in memory, uploaded with one `put`.
    Memory,
    /// The encoded bytes go out in parts of `part_bytes` while the file is
    /// written, through one multipart upload per file, by a caller that writes
    /// through `write_batch_async`; the footer is the last part. The default on
    /// `wasm32` ([`Staging::DEFAULT_PART_BYTES`]).
    ///
    /// A caller that writes through the synchronous `write_batch` gets no parts
    /// until the file is finished, and so holds the file whole as
    /// [`Staging::Memory`] does. S3 refuses a part other than the last below
    /// 5 MiB, so a store that is S3 wants `part_bytes` of at least that. Values
    /// below 1 are taken as 1.
    Parts {
        /// The size of every part but the last.
        part_bytes: usize,
    },
}

impl Staging {
    /// The part size of the `wasm32` default: 8 MiB, above S3's 5 MiB minimum
    /// and small beside a tab's memory pool.
    pub const DEFAULT_PART_BYTES: usize = 8 * 1024 * 1024;
}

// Not derived: natively the default is the unit `File`, but on `wasm32` it is
// `Parts`, which carries its part size, and `#[default]` takes only a unit variant.
#[allow(clippy::derivable_impls)]
impl Default for Staging {
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Staging::File
        }
        #[cfg(target_arch = "wasm32")]
        {
            Staging::Parts {
                part_bytes: Staging::DEFAULT_PART_BYTES,
            }
        }
    }
}

/// The sink a Parquet writer encodes into.
#[derive(Debug)]
pub(crate) enum StageWriter {
    #[cfg(not(target_arch = "wasm32"))]
    File {
        file: std::io::BufWriter<std::fs::File>,
        temp: tempfile::NamedTempFile,
    },
    Memory(Vec<u8>),
    Parts(PartsSink),
}

/// A [`Staging::Parts`] file being written: the bytes not yet sent, and the
/// upload the sent ones went to.
pub(crate) struct PartsSink {
    buffer: Vec<u8>,
    part_bytes: usize,
    /// Opened by the first part sent; `None` until a part is sent.
    upload: Option<Box<dyn MultipartUpload>>,
    /// Bytes sent in parts so far.
    sent: u64,
    /// Parts sent so far.
    parts: u64,
    /// Set while bytes taken from `buffer` are not yet acknowledged as a part:
    /// a [`Self::put_full_parts`] cancelled mid-await (its future dropped) has
    /// lost them, so the file can only be aborted, and finishing it is refused.
    cancelled: bool,
}

impl Drop for PartsSink {
    /// A file dropped with its upload open (a write that failed or was
    /// abandoned before its file was finished) has its upload aborted in the
    /// background where there is a tokio runtime to run the abort on; see
    /// [`abort_in_background`].
    fn drop(&mut self) {
        if let Some(upload) = self.upload.take() {
            abort_in_background(upload);
        }
    }
}

/// Abort `upload`, which its owner dropped before completing or aborting it. A
/// drop cannot await, so the abort runs as a task on the current tokio
/// runtime, best effort. With no runtime (a browser) it is only dropped: there
/// is nothing to spawn on, and an OPFS upload removes its temporary file when it
/// drops. An S3 upload left so keeps its parts until a bucket lifecycle rule
/// clears them. The awaited forms (`TableWriteSession::abort`, and every error
/// path of a write that can await) abort explicitly and do not rely on this.
fn abort_in_background(mut upload: Box<dyn MultipartUpload>) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => {
            runtime.spawn(async move {
                if let Err(error) = upload.abort().await {
                    tracing::warn!(error = %error, "failed to abort a dropped data file's multipart upload");
                }
            });
        },
        Err(_) => tracing::debug!(
            "a data file's multipart upload was dropped with no runtime to abort it on"
        ),
    }
}

fn cancelled_part() -> crate::error::DuckLakeError {
    crate::error::DuckLakeError::Internal(
        "a data file's part upload was cancelled before it completed; the file can only be aborted"
            .to_string(),
    )
}

impl std::fmt::Debug for PartsSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartsSink")
            .field("buffered", &self.buffer.len())
            .field("part_bytes", &self.part_bytes)
            .field("uploading", &self.upload.is_some())
            .field("sent", &self.sent)
            .field("parts", &self.parts)
            .field("cancelled", &self.cancelled)
            .finish()
    }
}

impl PartsSink {
    /// Send every whole part the buffer holds, opening the upload at the first.
    ///
    /// Not cancel-safe, and it says so: the bytes leave the buffer before their
    /// part is acknowledged. A call whose future is dropped mid-await marks the
    /// file `cancelled`, and its finish is then refused; the file must be
    /// aborted.
    async fn put_full_parts(
        &mut self,
        object_store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
    ) -> Result<()> {
        if self.buffer.len() < self.part_bytes {
            return Ok(());
        }
        if self.cancelled {
            return Err(cancelled_part());
        }
        let whole = self.buffer.len() - self.buffer.len() % self.part_bytes;
        let rest = self.buffer.split_off(whole);
        let full = Bytes::from(std::mem::replace(&mut self.buffer, rest));
        self.cancelled = true;
        self.put(object_store, object_path, full).await?;
        self.cancelled = false;
        Ok(())
    }

    /// The finished file, unless a part was lost to a cancelled
    /// [`Self::put_full_parts`].
    fn into_staged(mut self, metadata: Option<Arc<ParquetMetaData>>) -> Result<Staged> {
        if self.cancelled {
            return Err(cancelled_part());
        }
        Ok(Staged::Parts(StagedParts {
            tail: Bytes::from(std::mem::take(&mut self.buffer)),
            part_bytes: self.part_bytes,
            upload: self.upload.take(),
            sent: self.sent,
            parts: self.parts,
            metadata,
        }))
    }

    /// Whether a part was lost to a cancelled [`Self::put_full_parts`].
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Send `bytes` in parts of `part_bytes` (the last may be shorter).
    async fn put(
        &mut self,
        object_store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
        bytes: Bytes,
    ) -> Result<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            let end = (offset + self.part_bytes).min(bytes.len());
            let part = bytes.slice(offset..end);
            if self.upload.is_none() {
                self.upload = Some(object_store.put_multipart(object_path).await?);
            }
            let Some(upload) = self.upload.as_mut() else {
                // Unreachable: set just above.
                return Err(crate::error::DuckLakeError::Internal(
                    "a multipart upload was not opened".to_string(),
                ));
            };
            if let Err(error) = upload.put_part(PutPayload::from(part)).await {
                self.abort().await;
                return Err(error.into());
            }
            self.sent += (end - offset) as u64;
            self.parts += 1;
            offset = end;
        }
        Ok(())
    }

    /// Abort the upload, if one was opened. Best effort: a failure is logged,
    /// since the caller is already failing or abandoning the write.
    async fn abort(&mut self) {
        if let Some(mut upload) = self.upload.take()
            && let Err(error) = upload.abort().await
        {
            tracing::warn!(error = %error, "failed to abort a data file's multipart upload");
        }
    }
}

impl StageWriter {
    pub(crate) fn new(staging: Staging) -> std::io::Result<Self> {
        Ok(match staging {
            #[cfg(not(target_arch = "wasm32"))]
            Staging::File => {
                let temp = tempfile::NamedTempFile::new()?;
                let file = std::io::BufWriter::new(temp.reopen()?);
                StageWriter::File {
                    file,
                    temp,
                }
            },
            Staging::Memory => StageWriter::Memory(Vec::new()),
            Staging::Parts {
                part_bytes,
            } => StageWriter::Parts(PartsSink {
                buffer: Vec::new(),
                part_bytes: part_bytes.max(1),
                upload: None,
                sent: 0,
                parts: 0,
                cancelled: false,
            }),
        })
    }

    /// Send every whole part this file holds to `object_path`, awaiting each
    /// inline (no task is spawned). A no-op for a staging other than
    /// [`Staging::Parts`]. A failed part aborts the upload.
    pub(crate) async fn put_full_parts(
        &mut self,
        object_store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
    ) -> Result<()> {
        match self {
            StageWriter::Parts(sink) => sink.put_full_parts(object_store, object_path).await,
            _ => Ok(()),
        }
    }

    /// Whether a [`Staging::Parts`] file lost a part to a cancelled send, so it
    /// can only be aborted.
    pub(crate) fn is_cancelled(&self) -> bool {
        matches!(self, StageWriter::Parts(sink) if sink.is_cancelled())
    }

    /// The bytes this staging holds in memory: what a buffer has allocated
    /// (the whole file so far for [`Staging::Memory`], the bytes not yet sent
    /// for [`Staging::Parts`]), and a temporary file's write buffer.
    pub(crate) fn memory_size(&self) -> usize {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            StageWriter::File {
                file,
                ..
            } => file.capacity(),
            StageWriter::Memory(bytes) => bytes.capacity(),
            StageWriter::Parts(sink) => sink.buffer.capacity(),
        }
    }

    /// Mark a [`Staging::Parts`] file as having lost a part to a cancelled send:
    /// a unit test's way to make its finish fail.
    #[cfg(test)]
    pub(crate) fn cancel_for_test(&mut self) {
        if let StageWriter::Parts(sink) = self {
            sink.cancelled = true;
        }
    }

    /// Abort the multipart upload a [`Staging::Parts`] file opened, if any.
    pub(crate) async fn abort(&mut self) {
        if let StageWriter::Parts(sink) = self {
            sink.abort().await;
        }
    }

    /// [`Self::finish`], keeping the footer's `metadata` for a staging whose
    /// bytes may already be gone ([`Staging::Parts`]): its statistics are taken
    /// from it rather than from a footer read back.
    pub(crate) fn finish_with_metadata(self, metadata: ParquetMetaData) -> Result<Staged> {
        match self {
            StageWriter::Parts(sink) => sink.into_staged(Some(Arc::new(metadata))),
            other => other.finish(),
        }
    }

    /// Everything written so far, complete: a file's buffer is flushed and its
    /// descriptor released, so holding many finished files costs no descriptors.
    pub(crate) fn finish(self) -> Result<Staged> {
        Ok(match self {
            #[cfg(not(target_arch = "wasm32"))]
            StageWriter::File {
                file,
                temp,
            } => {
                file.into_inner()
                    .map_err(|e| crate::error::DuckLakeError::Io(e.into_error()))?;
                Staged::File(temp.into_temp_path())
            },
            StageWriter::Memory(bytes) => Staged::Memory(Bytes::from(bytes)),
            StageWriter::Parts(sink) => sink.into_staged(None)?,
        })
    }
}

impl Default for StageWriter {
    /// An empty in-memory sink: what a finished writer is left holding once its
    /// own sink is taken out of it.
    fn default() -> Self {
        StageWriter::Memory(Vec::new())
    }
}

impl Write for StageWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            StageWriter::File {
                file,
                ..
            } => file.write(buf),
            StageWriter::Memory(bytes) => bytes.write(buf),
            StageWriter::Parts(sink) => sink.buffer.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            StageWriter::File {
                file,
                ..
            } => file.flush(),
            StageWriter::Memory(_) | StageWriter::Parts(_) => Ok(()),
        }
    }
}

/// A finished Parquet file awaiting upload. A file's local copy is deleted when
/// this drops.
#[derive(Debug)]
pub(crate) enum Staged {
    #[cfg(not(target_arch = "wasm32"))]
    File(tempfile::TempPath),
    Memory(Bytes),
    Parts(StagedParts),
}

/// A finished [`Staging::Parts`] file: the parts already sent, and the tail (at
/// least the footer) still to send.
pub(crate) struct StagedParts {
    tail: Bytes,
    part_bytes: usize,
    /// The upload the sent parts went to: `None` when no part was sent (the
    /// whole file is `tail`), and once it is complete.
    upload: Option<Box<dyn MultipartUpload>>,
    /// Bytes sent in parts before the file was finished; zero means `tail` is
    /// the whole file.
    sent: u64,
    parts: u64,
    /// The footer as the writer returned it, when the file was finished
    /// through [`StageWriter::finish_with_metadata`].
    metadata: Option<Arc<ParquetMetaData>>,
}

impl Drop for StagedParts {
    /// A finished file dropped before its upload completed (a write abandoned
    /// with files still queued) has its upload aborted in the background where
    /// there is a runtime; see [`abort_in_background`].
    fn drop(&mut self) {
        if let Some(upload) = self.upload.take() {
            abort_in_background(upload);
        }
    }
}

impl std::fmt::Debug for StagedParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedParts")
            .field("tail", &self.tail.len())
            .field("part_bytes", &self.part_bytes)
            .field("uploading", &self.upload.is_some())
            .field("sent", &self.sent)
            .field("parts", &self.parts)
            .field("metadata", &self.metadata.is_some())
            .finish()
    }
}

impl Staged {
    /// The file's size in bytes.
    pub(crate) fn len(&self) -> Result<u64> {
        Ok(match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(path) => std::fs::metadata(path)?.len(),
            Staged::Memory(bytes) => bytes.len() as u64,
            Staged::Parts(parts) => parts.sent + parts.tail.len() as u64,
        })
    }

    /// The bytes this finished file holds in memory until it is uploaded: all
    /// of it for [`Staging::Memory`], its unsent tail for [`Staging::Parts`],
    /// none for a temporary file.
    pub(crate) fn memory_size(&self) -> usize {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(_) => 0,
            Staged::Memory(bytes) => bytes.len(),
            Staged::Parts(parts) => parts.tail.len(),
        }
    }

    /// How many parts were sent before the file was finished: zero for a
    /// staging other than [`Staging::Parts`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn parts_sent(&self) -> u64 {
        match self {
            Staged::Parts(parts) => parts.parts,
            _ => 0,
        }
    }

    /// The Parquet footer's Thrift metadata length, from the file's last 8 bytes.
    pub(crate) fn footer_size(&self) -> Result<i64> {
        let mut tail = [0u8; 8];
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(path) => {
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(path)?;
                if file.metadata()?.len() < 8 {
                    return Err(too_small());
                }
                file.seek(SeekFrom::End(-8))?;
                file.read_exact(&mut tail)?;
            },
            // The footer is written after the last part is sent, so the tail
            // always ends with the whole 8-byte trailer.
            Staged::Memory(bytes)
            | Staged::Parts(StagedParts {
                tail: bytes,
                ..
            }) => {
                if bytes.len() < 8 {
                    return Err(too_small());
                }
                tail.copy_from_slice(&bytes[bytes.len() - 8..]);
            },
        }
        footer_size_from_tail(&tail)
    }

    /// Per-column statistics from the footer this crate just wrote.
    pub(crate) fn own_column_stats(
        &self,
        column_ids: &[i64],
        row_count: i64,
        leaves: &WrittenLeafStats,
    ) -> Vec<ColumnStat> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(path) => crate::stats_collect::collect_own_written_column_stats(
                path, column_ids, row_count, leaves,
            ),
            Staged::Memory(bytes) => crate::stats_collect::collect_own_written_column_stats_from(
                bytes.clone(),
                "<in-memory staging>",
                column_ids,
                row_count,
                leaves,
            ),
            // Whether a part was sent, not whether the upload is still held: an
            // upload is taken once it completes, and the tail alone is then not
            // the file.
            Staged::Parts(parts) => match (&parts.metadata, parts.sent == 0) {
                (Some(metadata), _) => {
                    crate::stats_collect::collect_own_written_column_stats_from_metadata(
                        Arc::clone(metadata),
                        "<part-staged file>",
                        column_ids,
                        row_count,
                        leaves,
                    )
                },
                // Nothing sent: the tail is the whole file.
                (None, true) => crate::stats_collect::collect_own_written_column_stats_from(
                    parts.tail.clone(),
                    "<part-staged file>",
                    column_ids,
                    row_count,
                    leaves,
                ),
                (None, false) => {
                    tracing::warn!(
                        "a part-staged file was finished without its footer metadata; \
                         writing it without column stats"
                    );
                    Vec::new()
                },
            },
        }
    }

    /// Store the file at `object_path`.
    ///
    /// A file streams through `object_store`'s buffered writer, which switches to
    /// a multipart upload past its buffer, so there is no single-`PUT` ceiling. On
    /// a copy failure the multipart upload is aborted, so no partial object is
    /// left; on a failure to complete it deliberately is not (see
    /// `stream_to_upload`). Memory is one `put`, which leaves either the whole
    /// object or none. Parts sends the rest of the file and completes its upload,
    /// or, when no part was sent, is one `put`; a failed part aborts the upload,
    /// a failed `complete` does not (the object may exist; the caller removes it).
    pub(crate) async fn upload(
        &mut self,
        object_store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
    ) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(path) => {
                let local = tokio::fs::File::open(path).await?;
                let mut reader = tokio::io::BufReader::new(local);
                let mut upload = object_store::buffered::BufWriter::new(
                    Arc::clone(object_store),
                    object_path.clone(),
                );
                stream_to_upload(&mut reader, &mut upload).await
            },
            Staged::Memory(bytes) => object_store
                .put(object_path, PutPayload::from(bytes.clone()))
                .await
                .map(|_| ())
                .map_err(std::io::Error::other),
            Staged::Parts(parts) => parts
                .upload(object_store, object_path)
                .await
                .map_err(std::io::Error::other),
        }
    }

    /// Abort the multipart upload a [`Staging::Parts`] file opened, if any: for a
    /// finished file abandoned before [`Self::upload`].
    pub(crate) async fn abort(&mut self) {
        if let Staged::Parts(parts) = self
            && let Some(mut upload) = parts.upload.take()
            && let Err(error) = upload.abort().await
        {
            tracing::warn!(error = %error, "failed to abort a data file's multipart upload");
        }
    }
}

impl StagedParts {
    async fn upload(
        &mut self,
        object_store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
    ) -> Result<()> {
        if self.sent == 0 {
            object_store
                .put(object_path, PutPayload::from(self.tail.clone()))
                .await?;
            return Ok(());
        }
        if self.upload.is_none() {
            return Err(crate::error::DuckLakeError::Internal(
                "a part-staged data file was uploaded twice".to_string(),
            ));
        }
        let mut sink = PartsSink {
            buffer: Vec::new(),
            part_bytes: self.part_bytes,
            upload: self.upload.take(),
            sent: self.sent,
            parts: self.parts,
            cancelled: false,
        };
        sink.put(object_store, object_path, self.tail.clone())
            .await?;
        let Some(mut upload) = sink.upload.take() else {
            // Unreachable: `put` keeps the upload unless it failed, and then
            // returned the error above.
            return Err(crate::error::DuckLakeError::Internal(
                "a data file's multipart upload was lost".to_string(),
            ));
        };
        // `sent` and `parts` stay what was sent before the file was finished:
        // `len` is `sent` plus the tail.
        upload.complete().await?;
        Ok(())
    }
}

/// Stream a reader to object storage and finalise the upload.
///
/// The abort decision lives here rather than at each call site, because only ONE
/// of the two failure modes may be aborted and getting it wrong is a panic, not a
/// wrong result: `BufWriter::abort` panics if the writer has already been shut
/// down, and a failing `shutdown` leaves it in exactly that state. So aborting
/// after a flush failure turns a recoverable upload error into a panicking task.
/// Aborting after a copy failure is both safe and necessary — it releases the
/// multipart upload instead of stranding it.
#[cfg(not(target_arch = "wasm32"))]
async fn stream_to_upload<R>(
    reader: &mut R,
    upload: &mut object_store::buffered::BufWriter,
) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin + ?Sized,
{
    use tokio::io::AsyncWriteExt;
    if let Err(e) = tokio::io::copy(reader, upload).await {
        let _ = upload.abort().await;
        return Err(e);
    }
    upload.shutdown().await
}

fn too_small() -> crate::error::DuckLakeError {
    crate::error::DuckLakeError::Internal("Invalid Parquet file: too small".to_string())
}

/// The Thrift metadata length from a Parquet file's 8-byte trailer (length, then
/// `PAR1`). The trailer itself is excluded. Stored as the nullable `footer_size`
/// hint in the catalog; readers fall back to a standard footer read without it.
pub(crate) fn footer_size_from_tail(buffer: &[u8]) -> Result<i64> {
    if buffer.len() < 8 {
        return Err(too_small());
    }
    let footer_bytes = &buffer[buffer.len() - 8..];
    if &footer_bytes[4..8] != b"PAR1" {
        return Err(crate::error::DuckLakeError::Internal(
            "Invalid Parquet file: missing PAR1 magic".to_string(),
        ));
    }
    let metadata_len =
        u32::from_le_bytes([footer_bytes[0], footer_bytes[1], footer_bytes[2], footer_bytes[3]]);
    Ok(i64::from(metadata_len))
}

#[cfg(all(test, feature = "write"))]
mod tests {
    use std::collections::HashMap;

    use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use arrow::record_batch::RecordBatch;
    use object_store::memory::InMemory;
    use parquet::file::properties::WriterProperties;

    use super::*;
    use crate::write_encoding::RowGroupSampledWriter;

    fn id(field: Field, id: i64) -> Field {
        field.with_metadata(HashMap::from([(
            "PARQUET:field_id".to_string(),
            id.to_string(),
        )]))
    }

    fn schema() -> Arc<Schema> {
        let inner = Fields::from(vec![
            id(Field::new("a", DataType::Int64, true), 4),
            id(Field::new("b", DataType::Utf8, true), 5),
        ]);
        Arc::new(Schema::new(vec![
            id(Field::new("id", DataType::Int64, false), 1),
            id(Field::new("name", DataType::Utf8, true), 2),
            id(Field::new("s", DataType::Struct(inner.clone()), true), 3),
            id(Field::new("v", DataType::Float64, true), 6),
        ]))
    }

    fn batch(lo: i64) -> RecordBatch {
        let ids: Vec<i64> = (lo..lo + 1000).collect();
        let names: Vec<Option<String>> = ids
            .iter()
            .map(|i| (i % 7 != 0).then(|| format!("name-{}", i * 31 % 9973)))
            .collect();
        let a: ArrayRef = Arc::new(Int64Array::from_iter(
            ids.iter().map(|i| (i % 5 != 0).then_some(-i)),
        ));
        let b: ArrayRef = Arc::new(StringArray::from_iter(
            ids.iter()
                .map(|i| (i % 3 != 0).then(|| format!("b{}", i % 101))),
        ));
        let DataType::Struct(fields) = schema().field(2).data_type().clone() else {
            unreachable!()
        };
        let s = StructArray::new(fields, vec![a, b], None);
        let v = Float64Array::from_iter(ids.iter().map(|i| match i {
            i if i % 13 == 0 => None,
            2_500 => Some(f64::NAN),
            i => Some(*i as f64 / 3.0),
        }));
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(s),
                Arc::new(v),
            ],
        )
        .unwrap()
    }

    /// A part-staged file sends its parts while it is written, not at the end;
    /// the object is exactly the bytes the writer produced; and the statistics
    /// taken from the footer the writer returned equal those of the footer read
    /// back from the stored object, nested fields and NaN included.
    #[tokio::test]
    async fn parts_go_out_while_writing_and_stats_equal_the_footer_read_back() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = ObjectPath::from("t/file.parquet");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(500))
            .build();
        let mut writer = RowGroupSampledWriter::new(
            StageWriter::new(Staging::Parts {
                part_bytes: 4 * 1024,
            })
            .unwrap(),
            schema(),
            props,
        );
        let mut leaves = crate::stats_collect::WrittenLeafStats::default();
        let mut rows = 0;
        let mut sent_while_writing = Vec::new();
        for b in 0..20 {
            let batch = batch(b * 1000);
            leaves.accumulate(&batch, batch.num_columns());
            writer.write(&batch).unwrap();
            rows += batch.num_rows() as i64;
            let sink = writer.sink_mut().unwrap();
            sink.put_full_parts(&store, &path).await.unwrap();
            let StageWriter::Parts(parts) = sink else {
                unreachable!()
            };
            assert!(
                parts.buffer.len() < parts.part_bytes,
                "only a part's tail is held"
            );
            sent_while_writing.push(parts.parts);
        }
        assert!(
            store.head(&path).await.is_err(),
            "no object before the upload completes"
        );
        let (sink, metadata) = writer.finish().unwrap();
        let mut staged = sink.finish_with_metadata(metadata).unwrap();
        let parts = staged.parts_sent();
        assert!(parts >= 10, "{parts} parts sent before finish");
        assert!(
            sent_while_writing
                .windows(2)
                .filter(|w| w[1] > w[0])
                .count()
                >= 10,
            "parts went out batch by batch: {sent_while_writing:?}"
        );

        let column_ids = [1, 2, 3, 6];
        let from_metadata = staged.own_column_stats(&column_ids, rows, &leaves);
        let size = staged.len().unwrap();
        let footer = staged.footer_size().unwrap();
        staged.upload(&store, &path).await.unwrap();

        let stored = store.get(&path).await.unwrap().bytes().await.unwrap();
        assert_eq!(stored.len() as u64, size);
        assert_eq!(footer_size_from_tail(&stored).unwrap(), footer);
        let read_back = crate::stats_collect::collect_own_written_column_stats_from(
            stored,
            "read back",
            &column_ids,
            rows,
            &leaves,
        );
        assert_eq!(
            from_metadata
                .iter()
                .map(|s| s.column_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5, 6],
            "a stat per scalar column and nested field, by field id"
        );
        assert!(
            from_metadata.iter().any(|s| s.contains_nan == Some(true)),
            "{from_metadata:?}"
        );
        assert_eq!(format!("{from_metadata:?}"), format!("{read_back:?}"));
    }
}
