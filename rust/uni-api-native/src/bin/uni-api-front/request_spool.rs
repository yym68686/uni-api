use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::StatusCode;
use bytes::Bytes;
use futures_util::Stream;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio_util::io::ReaderStream;

use crate::idempotency::{RequestHasher, RequestIdentity};
use crate::resources::{CapacityFailure, ResourceGovernor, ResourceKind};

static NEXT_SPOOL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Default)]
pub struct SpoolObservation {
    pub body_bytes: u64,
    pub memory_peak_bytes: u64,
    pub local_disk_bytes: u64,
    pub local_free_bytes_at_start: u64,
    pub local_writable_bytes_at_start: u64,
    pub local_free_inodes_at_start: u64,
    pub local_writable_inodes_at_start: u64,
    pub resource_wait_ms: u64,
    pub final_tier: &'static str,
    pub failure_resource: Option<&'static str>,
}

#[derive(Debug)]
pub struct SpoolFailure {
    pub status: StatusCode,
    pub message: String,
    pub resource: Option<ResourceKind>,
    pub retry_after: bool,
    pub observation: SpoolObservation,
}

impl SpoolFailure {
    fn capacity(failure: CapacityFailure, mut observation: SpoolObservation) -> Self {
        observation.resource_wait_ms = observation
            .resource_wait_ms
            .saturating_add(duration_ms(failure.waited));
        observation.failure_resource = Some(failure.resource.as_str());
        Self {
            status: failure.resource.exhausted_status(),
            message: format!(
                "request spool resource wait timed out: {}",
                failure.resource.as_str()
            ),
            resource: Some(failure.resource),
            retry_after: true,
            observation,
        }
    }

    fn io(message: impl Into<String>, mut observation: SpoolObservation) -> Self {
        observation.failure_resource = Some(ResourceKind::LocalDisk.as_str());
        Self {
            status: StatusCode::INSUFFICIENT_STORAGE,
            message: message.into(),
            resource: Some(ResourceKind::LocalDisk),
            retry_after: true,
            observation,
        }
    }
}

#[derive(Clone)]
pub struct SpoolManager {
    root: Arc<PathBuf>,
    governor: ResourceGovernor,
}

impl SpoolManager {
    pub fn new(governor: ResourceGovernor) -> Result<Self, String> {
        let root = PathBuf::from(
            std::env::var("RUST_REQUEST_SPOOL_DIRECTORY")
                .unwrap_or_else(|_| "/tmp/uni-api-request-spool".to_owned()),
        );
        std::fs::create_dir_all(&root)
            .map_err(|error| format!("create request spool directory: {error}"))?;
        Ok(Self {
            root: Arc::new(root),
            governor,
        })
    }

    pub async fn begin(
        &self,
        request_hasher: Option<RequestHasher>,
        content_length: Option<u64>,
        initial_wait: Duration,
    ) -> Result<RequestSpoolWriter, SpoolFailure> {
        let mut observation = SpoolObservation {
            resource_wait_ms: duration_ms(initial_wait),
            ..SpoolObservation::default()
        };
        let (wait, capacity) = self
            .governor
            .wait_for_local_capacity(&self.root, content_length.unwrap_or(0))
            .await
            .map_err(|failure| SpoolFailure::capacity(failure, observation.clone()))?;
        observation.resource_wait_ms = observation
            .resource_wait_ms
            .saturating_add(duration_ms(wait.waited));
        observation.local_free_bytes_at_start = capacity.free_bytes;
        observation.local_writable_bytes_at_start = capacity.writable_bytes;
        observation.local_free_inodes_at_start = capacity.free_inodes;
        observation.local_writable_inodes_at_start = capacity.writable_inodes;
        let writer = self.create_local_writer().await.map_err(|error| {
            SpoolFailure::io(
                format!("create local request spool: {error}"),
                observation.clone(),
            )
        })?;
        Ok(RequestSpoolWriter {
            manager: self.clone(),
            request_hasher,
            writer: Some(writer),
            observation,
        })
    }

    async fn create_local_writer(&self) -> std::io::Result<LocalWriter> {
        for _ in 0..16 {
            let path = self.root.join(unique_name());
            match OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .await
            {
                Ok(file) => return Ok(LocalWriter { file, path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique spool path",
        ))
    }
}

pub struct RequestSpoolWriter {
    manager: SpoolManager,
    request_hasher: Option<RequestHasher>,
    writer: Option<LocalWriter>,
    observation: SpoolObservation,
}

impl RequestSpoolWriter {
    pub async fn append(&mut self, chunk: Bytes) -> Result<(), SpoolFailure> {
        if chunk.is_empty() {
            return Ok(());
        }
        self.observation.body_bytes = self
            .observation
            .body_bytes
            .saturating_add(chunk.len() as u64);
        self.observation.memory_peak_bytes =
            self.observation.memory_peak_bytes.max(chunk.len() as u64);
        let (wait, _) = self
            .manager
            .governor
            .wait_for_local_capacity(&self.manager.root, chunk.len() as u64)
            .await
            .map_err(|failure| SpoolFailure::capacity(failure, self.observation.clone()))?;
        self.observation.resource_wait_ms = self
            .observation
            .resource_wait_ms
            .saturating_add(duration_ms(wait.waited));
        self.writer
            .as_mut()
            .expect("request spool writer missing")
            .file
            .write_all(&chunk)
            .await
            .map_err(|error| {
                SpoolFailure::io(
                    format!("write local request spool: {error}"),
                    self.observation.clone(),
                )
            })?;
        if let Some(request_hasher) = self.request_hasher.as_mut() {
            request_hasher.update(&chunk);
        }
        self.observation.local_disk_bytes = self
            .observation
            .local_disk_bytes
            .saturating_add(chunk.len() as u64);
        Ok(())
    }

    pub async fn finish(mut self) -> Result<RequestSpool, SpoolFailure> {
        let mut writer = self.writer.take().expect("request spool writer missing");
        writer.file.flush().await.map_err(|error| {
            SpoolFailure::io(
                format!("flush local request spool: {error}"),
                self.observation.clone(),
            )
        })?;
        drop(writer.file);
        self.observation.final_tier = "local_disk";
        Ok(RequestSpool {
            identity: self.request_hasher.take().map(RequestHasher::finish),
            observation: self.observation.clone(),
            storage: StoredBody::local(writer.path),
        })
    }
}

impl Drop for RequestSpoolWriter {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            drop(writer.file);
            let _ = std::fs::remove_file(writer.path);
        }
    }
}

pub struct RequestSpool {
    pub identity: Option<RequestIdentity>,
    pub observation: SpoolObservation,
    pub storage: StoredBody,
}

#[derive(Clone)]
pub struct StoredBody {
    path: PathBuf,
    cleanup: Arc<LocalCleanup>,
}

pub struct MultipartFile {
    pub filename: String,
    pub content_type: Option<String>,
    pub storage: StoredBody,
    pub bytes: u64,
}

impl StoredBody {
    fn local(path: PathBuf) -> Self {
        Self {
            cleanup: Arc::new(LocalCleanup { path: path.clone() }),
            path,
        }
    }

    pub async fn into_body(self, observation: &SpoolObservation) -> Result<Body, SpoolFailure> {
        let file = File::open(&self.path).await.map_err(|error| {
            SpoolFailure::io(
                format!("open local request spool for replay: {error}"),
                observation.clone(),
            )
        })?;
        Ok(Body::from_stream(LocalBodyStream {
            inner: ReaderStream::new(file),
            _cleanup: self.cleanup,
        }))
    }

    pub async fn parse_json(&self) -> Result<serde_json::Value, String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path)
                .map_err(|error| format!("open local request spool for JSON parse: {error}"))?;
            serde_json::from_reader(std::io::BufReader::new(file))
                .map_err(|error| format!("invalid JSON request body: {error}"))
        })
        .await
        .map_err(|error| format!("request JSON parser task failed: {error}"))?
    }

    pub fn clone_for_replay(&self) -> Self {
        self.clone()
    }

    pub async fn multipart_text_field(
        &self,
        content_type: &str,
        field_name: &str,
        max_value_bytes: usize,
    ) -> Result<Option<String>, String> {
        let boundary = multer::parse_boundary(content_type)
            .map_err(|error| format!("multipart request is missing a valid boundary: {error}"))?;
        let file = File::open(&self.path)
            .await
            .map_err(|error| format!("open multipart request spool: {error}"))?;
        let mut multipart = multer::Multipart::new(ReaderStream::new(file), boundary);
        while let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|error| format!("parse multipart request: {error}"))?
        {
            if field.name() != Some(field_name) {
                continue;
            }
            let mut value = Vec::new();
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|error| format!("read multipart field {field_name}: {error}"))?
            {
                if value.len().saturating_add(chunk.len()) > max_value_bytes {
                    return Err(format!(
                        "multipart field {field_name} exceeds {max_value_bytes} bytes"
                    ));
                }
                value.extend_from_slice(&chunk);
            }
            return std::str::from_utf8(&value)
                .map(|value| Some(value.trim().to_owned()))
                .map_err(|error| format!("multipart field {field_name} is not UTF-8: {error}"));
        }
        Ok(None)
    }

    pub async fn multipart_file(
        &self,
        content_type: &str,
        field_name: &str,
    ) -> Result<Option<MultipartFile>, String> {
        let boundary = multer::parse_boundary(content_type)
            .map_err(|error| format!("multipart request is missing a valid boundary: {error}"))?;
        let file = File::open(&self.path)
            .await
            .map_err(|error| format!("open multipart request spool: {error}"))?;
        let mut multipart = multer::Multipart::new(ReaderStream::new(file), boundary);
        while let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|error| format!("parse multipart request: {error}"))?
        {
            if field.name() != Some(field_name) {
                continue;
            }
            let filename = field
                .file_name()
                .filter(|value| !value.is_empty())
                .unwrap_or("audio.bin")
                .to_owned();
            let content_type = field.content_type().map(ToString::to_string);
            let output_path = self.path.with_file_name(unique_name());
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&output_path)
                .await
                .map_err(|error| format!("create multipart file spool: {error}"))?;
            let output_storage = StoredBody::local(output_path);
            let mut bytes = 0u64;
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|error| format!("read multipart file {field_name}: {error}"))?
            {
                bytes = bytes.saturating_add(chunk.len() as u64);
                output
                    .write_all(&chunk)
                    .await
                    .map_err(|error| format!("write multipart file spool: {error}"))?;
            }
            output
                .flush()
                .await
                .map_err(|error| format!("flush multipart file spool: {error}"))?;
            drop(output);
            return Ok(Some(MultipartFile {
                filename,
                content_type,
                storage: output_storage,
                bytes,
            }));
        }
        Ok(None)
    }
}

struct LocalWriter {
    file: File,
    path: PathBuf,
}

struct LocalCleanup {
    path: PathBuf,
}

impl Drop for LocalCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct LocalBodyStream {
    inner: ReaderStream<File>,
    _cleanup: Arc<LocalCleanup>,
}

impl Stream for LocalBodyStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

fn unique_name() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = NEXT_SPOOL_ID.fetch_add(1, Ordering::Relaxed);
    format!("request-{}-{now}-{id}.spool", std::process::id())
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::{HeaderMap, HeaderValue, Method, Uri};

    fn request_hasher() -> RequestHasher {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        RequestHasher::new(
            &Method::POST,
            &Uri::from_static("/v1/responses"),
            &headers,
            "spool-test",
        )
        .unwrap()
    }

    #[tokio::test]
    async fn local_spool_is_streamed_without_whole_body_buffering() {
        let manager = SpoolManager::new(ResourceGovernor::unconstrained_for_test()).unwrap();
        let mut writer = manager
            .begin(Some(request_hasher()), Some(11), Duration::ZERO)
            .await
            .unwrap();
        writer.append(Bytes::from_static(b"hello ")).await.unwrap();
        writer.append(Bytes::from_static(b"world")).await.unwrap();
        let spool = writer.finish().await.unwrap();
        assert_eq!(spool.observation.body_bytes, 11);
        assert_eq!(spool.observation.local_disk_bytes, 11);
        assert_eq!(spool.observation.final_tier, "local_disk");
        let observation = spool.observation.clone();
        let body = spool.storage.into_body(&observation).await.unwrap();
        assert_eq!(
            to_bytes(body, 64).await.unwrap(),
            Bytes::from_static(b"hello world")
        );
    }

    #[tokio::test]
    async fn spool_has_no_single_request_byte_ceiling() {
        let manager = SpoolManager::new(ResourceGovernor::unconstrained_for_test()).unwrap();
        let mut writer = manager
            .begin(Some(request_hasher()), None, Duration::ZERO)
            .await
            .unwrap();
        let chunk = Bytes::from(vec![b'x'; 1024 * 1024]);
        for _ in 0..17 {
            writer.append(chunk.clone()).await.unwrap();
        }
        let spool = writer.finish().await.unwrap();
        assert_eq!(spool.observation.body_bytes, 17 * 1024 * 1024);
        assert_eq!(spool.observation.local_disk_bytes, 17 * 1024 * 1024);
    }
}
