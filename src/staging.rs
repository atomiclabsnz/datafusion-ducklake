//! Where a Parquet file waits between the synchronous writer that encodes it and
//! the asynchronous upload that stores it.
//!
//! [`Staging::File`] is a local temporary file: a multi-GB table is never held in
//! RAM, and the upload streams it out through a multipart upload. It is the
//! default wherever there is a filesystem.
//!
//! [`Staging::Memory`] holds the encoded bytes in memory and uploads them with a
//! single `put`. It is the only staging on `wasm32` (`wasm32-unknown-unknown` has
//! no filesystem: `std::fs` compiles there and fails every call), and it is
//! available everywhere so the same path can be tested natively. A staged file is
//! bounded by the writer's `target_file_size`, which is how a caller bounds the
//! memory this costs. The single `put` is also what keeps the upload free of
//! spawned tasks: `object_store`'s multipart writer spawns one per part, and a
//! browser has no tokio runtime to run them on.

use std::io::Write;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

use crate::Result;
use crate::metadata_writer::ColumnStat;
use crate::stats_collect::WrittenLeafStats;

/// How a writer stages each Parquet file before upload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Staging {
    /// A local temporary file, streamed out through a multipart upload.
    /// Not available on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    #[default]
    File,
    /// The encoded bytes in memory, uploaded with one `put`. The default on
    /// `wasm32`.
    #[cfg_attr(target_arch = "wasm32", default)]
    Memory,
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
        })
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
        })
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
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            StageWriter::File {
                file,
                ..
            } => file.flush(),
            StageWriter::Memory(_) => Ok(()),
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
}

impl Staged {
    /// The file's size in bytes.
    pub(crate) fn len(&self) -> Result<u64> {
        Ok(match self {
            #[cfg(not(target_arch = "wasm32"))]
            Staged::File(path) => std::fs::metadata(path)?.len(),
            Staged::Memory(bytes) => bytes.len() as u64,
        })
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
            Staged::Memory(bytes) => {
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
        }
    }

    /// Store the file at `object_path`.
    ///
    /// A file streams through `object_store`'s buffered writer, which switches to
    /// a multipart upload past its buffer, so there is no single-`PUT` ceiling. On
    /// a copy failure the multipart upload is aborted, so no partial object is
    /// left; on a failure to complete it deliberately is not (see
    /// `stream_to_upload`). Memory is one `put`, which leaves either the whole
    /// object or none.
    pub(crate) async fn upload(
        &self,
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
        }
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
