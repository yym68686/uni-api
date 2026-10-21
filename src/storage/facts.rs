//! Bounded, authenticated request-fact exporter. The queue is outside the
//! request critical path: full or unavailable storage drops facts with an
//! explicit counter, while serving remains available.

use reqwest::Client;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use url::Url;

const QUEUE_CAPACITY: usize = 4096;

const BATCH_SIZE: usize = 128;

const BATCH_WAIT: Duration = Duration::from_secs(2);

const UPLOAD_RETRIES: usize = 3;
const SPOOL_LIMIT: u64 = 64 * 1024 * 1024;
static DROPPED: AtomicU64 = AtomicU64::new(0);
static QUEUED: AtomicU64 = AtomicU64::new(0);
static EXPORT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct UploadConfig {
    client: Client,
    endpoint: String,
    token: String,
    instance: String,
    epoch: String,
    spool: PathBuf,
}

#[derive(Clone)]
pub struct FactWriter {
    sender: mpsc::Sender<Value>,
}

impl FactWriter {
    pub fn from_env() -> Option<Self> {
        let (endpoint, token) = ingest_target_from_env()?;
        let (sender, mut rx) = mpsc::channel(QUEUE_CAPACITY);
        let config = UploadConfig {
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(20))
                .build()
                .ok()?,
            endpoint,
            token,
            instance: instance_id(),
            epoch: new_event_id("exporter"),
            spool: PathBuf::from(
                std::env::var("FACTS_SPOOL_DIR")
                    .or_else(|_| std::env::var("FACTS_S3_SPOOL_DIR"))
                    .unwrap_or_else(|_| "./data/facts-spool".into()),
            ),
        };
        if std::fs::create_dir_all(&config.spool).is_err() {
            eprintln!("facts_ingest_init_failed reason=spool_unavailable");
            return None;
        }
        let retry_config = config.clone();
        tokio::spawn(async move {
            retry_spool(retry_config).await;
        });
        tokio::spawn(async move {
            while let Some(batch) = receive_batch(&mut rx, BATCH_WAIT).await {
                let Some((key, body)) = batch_payload(&config, &batch) else {
                    QUEUED.fetch_sub(batch.len() as u64, Ordering::Relaxed);
                    DROPPED.fetch_add(batch.len() as u64, Ordering::Relaxed);
                    continue;
                };
                let spool = persist_batch(&config, &key, &body).await;
                QUEUED.fetch_sub(batch.len() as u64, Ordering::Relaxed);
                if let Err(error) = upload_body_with_retry(&config, &key, &body).await {
                    if !spool {
                        DROPPED.fetch_add(batch.len() as u64, Ordering::Relaxed);
                    }
                    eprintln!(
                        "{{\"event_type\":\"facts_ingest_upload_error\",\"error\":{:?},\"spooled\":{}}}",
                        error,
                        spool
                    )
                } else if spool {
                    let _ = tokio::fs::remove_file(
                        config.spool.join(format!("{}.jsonl", hex_sha(&body))),
                    )
                    .await;
                }
            }
        });
        Some(Self { sender })
    }
    pub fn enqueue(&self, event: Value) {
        QUEUED.fetch_add(1, Ordering::Relaxed);
        if self.sender.try_send(event).is_err() {
            QUEUED.fetch_sub(1, Ordering::Relaxed);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            eprintln!("facts_ingest_queue_dropped record_type=request_fact");
        }
    }
}

async fn receive_batch(rx: &mut mpsc::Receiver<Value>, wait: Duration) -> Option<Vec<Value>> {
    // recv returns None only after all buffered facts are drained, even when
    // every sender has already closed. is_closed alone is not an empty check.
    let first = rx.recv().await?;
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    batch.push(first);
    let deadline = tokio::time::sleep(wait);
    tokio::pin!(deadline);
    while batch.len() < BATCH_SIZE {
        tokio::select! {
            value = rx.recv() => match value {
                Some(v) => batch.push(v),
                None => break,
            },
            _ = &mut deadline => break,
        }
    }
    Some(batch)
}

fn batch_payload(_c: &UploadConfig, batch: &[Value]) -> Option<(String, String)> {
    let body = batch
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .ok()?
        .join("\n")
        + "\n";
    let digest = hex_sha(&body);
    let key = digest;
    Some((key, body))
}

// Only a digest-matching commit acknowledgement permits deleting a batch.
// Redirects are disabled so the credential cannot leave its configured origin.
async fn upload_body(c: &UploadConfig, _key: &str, body: &str) -> Result<(), String> {
    if body.len() > 4 * 1024 * 1024 {
        return Err("fact batch too large".into());
    }
    let hash = hex_sha(body);
    let response = c
        .client
        .post(&c.endpoint)
        .bearer_auth(&c.token)
        .header("Content-Type", "application/x-ndjson")
        .header("X-Uni-Facts-Version", "1")
        .header("X-Uni-Facts-Instance", &c.instance)
        .header("X-Uni-Facts-Epoch", &c.epoch)
        .header(
            "X-Uni-Facts-Sequence",
            EXPORT_SEQUENCE.fetch_add(1, Ordering::Relaxed).to_string(),
        )
        .header(
            "X-Uni-Facts-Dropped",
            DROPPED.load(Ordering::Relaxed).to_string(),
        )
        .header(
            "X-Uni-Facts-Pending",
            QUEUED
                .load(Ordering::Relaxed)
                .saturating_add(
                    spool_pending(c)
                        .await
                        .saturating_sub(u64::from(!body.is_empty())),
                )
                .to_string(),
        )
        .body(body.to_owned())
        .send()
        .await
        .map_err(|_| "fact ingestion transport failed")?;
    if !response.status().is_success() {
        return Err(format!(
            "fact ingestion HTTP {}",
            response.status().as_u16()
        ));
    }
    if response.content_length().is_some_and(|n| n > 4096) {
        return Err("invalid fact acknowledgement".into());
    }
    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(part) = stream.next().await {
        let part = part.map_err(|_| "incomplete fact acknowledgement")?;
        if bytes.len() + part.len() > 4096 {
            return Err("invalid fact acknowledgement".into());
        }
        bytes.extend_from_slice(&part);
    }
    let ack: Value = serde_json::from_slice(&bytes).map_err(|_| "invalid fact acknowledgement")?;
    if ack["format"] != "facts-ack-v1" || ack["sha256"] != hash {
        return Err("fact acknowledgement mismatch".into());
    }
    Ok(())
}

fn ingest_target_from_env() -> Option<(String, String)> {
    let explicit = std::env::var("FACTS_INGEST_URL")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let endpoint = match explicit {
        Some(url) => url,
        None => {
            let restore = std::env::var("UNI_API_CONTROL_RESTORE_URL").ok()?;
            ingest_url(&restore)?
        }
    };
    let token = std::env::var("FACTS_INGEST_TOKEN")
        .or_else(|_| std::env::var("UNI_API_CONTROL_RESTORE_TOKEN"))
        .ok()?;
    let parsed = Url::parse(&endpoint).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || token.trim().is_empty()
    {
        return None;
    }
    Some((endpoint, token))
}

fn ingest_url(restore: &str) -> Option<String> {
    let mut url = Url::parse(restore).ok()?;
    let (prefix, id) = url.path().rsplit_once("/v1/runtime-restore/")?;
    if id.is_empty() || id.contains('/') {
        return None;
    }
    url.set_path(&format!("{prefix}/v1/runtime-facts/{id}"));
    Some(url.to_string())
}

async fn persist_batch(c: &UploadConfig, key: &str, body: &str) -> bool {
    if body.len() > 4 * 1024 * 1024 {
        return false;
    }
    if !reserve_spool(c, body.len() as u64).await {
        return false;
    }
    let digest = hex_sha(body);
    let target = c.spool.join(format!("{}.jsonl", digest));
    if tokio::fs::try_exists(&target).await.unwrap_or(false) {
        return true;
    }
    let pending = c.spool.join(format!("{}.pending", digest));
    if tokio::fs::write(&pending, body.as_bytes()).await.is_err() {
        eprintln!("facts_ingest_spool_write_failed digest={digest}");
        return false;
    }
    if let Ok(file) = tokio::fs::OpenOptions::new()
        .write(true)
        .open(&pending)
        .await
    {
        let _ = file.sync_all().await;
    }
    if tokio::fs::rename(&pending, &target).await.is_err() {
        eprintln!("facts_ingest_spool_commit_failed digest={digest}");
        return false;
    }
    let _ = key;
    true
}

async fn retry_spool(c: UploadConfig) {
    loop {
        if let Ok(mut entries) = tokio::fs::read_dir(&c.spool).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().and_then(|v| v.to_str()) != Some("jsonl") {
                    continue;
                }
                let Ok(body) = tokio::fs::read_to_string(&path).await else {
                    continue;
                };
                let Some(name) = path.file_stem().and_then(|v| v.to_str()) else {
                    continue;
                };
                let key = name.to_string();
                if upload_body_with_retry(&c, &key, &body).await.is_ok() {
                    let _ = tokio::fs::remove_file(path).await;
                }
            }
        }
        if QUEUED.load(Ordering::Relaxed) == 0 && spool_pending(&c).await == 0 {
            let _ = upload_body(&c, "", "").await;
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

async fn spool_pending(c: &UploadConfig) -> u64 {
    let Ok(mut entries) = tokio::fs::read_dir(&c.spool).await else {
        return 1;
    };
    let mut count = 0;
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.path().extension().and_then(|v| v.to_str()) == Some("jsonl") {
            count += 1;
        }
        if count >= 4096 {
            break;
        }
    }
    count
}

async fn reserve_spool(c: &UploadConfig, incoming: u64) -> bool {
    let Ok(mut entries) = tokio::fs::read_dir(&c.spool).await else {
        return false;
    };
    let mut files = Vec::new();
    let mut size = 0u64;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|v| v.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            return false;
        };
        size = size.saturating_add(meta.len());
        files.push((meta.modified().unwrap_or(UNIX_EPOCH), path, meta.len()));
        if files.len() >= 4096 {
            return false;
        }
    }
    files.sort_by_key(|row| row.0);
    for (_, path, bytes) in files {
        if size.saturating_add(incoming) <= SPOOL_LIMIT {
            break;
        }
        let Ok(body) = tokio::fs::read_to_string(&path).await else {
            return false;
        };
        let records = body.lines().filter(|line| !line.trim().is_empty()).count() as u64;
        if tokio::fs::remove_file(&path).await.is_err() {
            return false;
        }
        size = size.saturating_sub(bytes);
        DROPPED.fetch_add(records, Ordering::Relaxed);
        eprintln!("facts_ingest_spool_evicted bytes={bytes}");
    }
    size.saturating_add(incoming) <= SPOOL_LIMIT
}

async fn upload_body_with_retry(c: &UploadConfig, key: &str, body: &str) -> Result<(), String> {
    let mut last = String::from("fact ingestion failed");
    for attempt in 0..UPLOAD_RETRIES {
        match upload_body(c, key, body).await {
            Ok(()) => return Ok(()),
            Err(error) => {
                last = error;
                if attempt + 1 < UPLOAD_RETRIES {
                    tokio::time::sleep(Duration::from_millis(200 * (1u64 << attempt))).await;
                }
            }
        }
    }
    Err(last)
}

fn hex_sha(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

static WRITER: OnceLock<Option<FactWriter>> = OnceLock::new();

pub fn global() -> Option<FactWriter> {
    WRITER.get_or_init(FactWriter::from_env).clone()
}

pub fn request_event(s: &crate::storage::database::RequestStat) -> Value {
    let key = hex_sha(&s.api_key);
    let at = now_ms();
    json!({"schema":1,"kind":"request","event_id":new_event_id("request"),"instance_id":instance_id(),"at_ms":at,"request_id":s.request_id,"trace_id":s.trace_id,"key_id":format!("key-{}",key),"endpoint":s.endpoint,"provider":s.provider,"model":s.model,"upstream_model":s.upstream_model,"stream":s.stream,"status":s.status,"outcome":if s.is_flagged{"failed"}else{"success"},"terminal_kind":s.terminal_kind,"failure_reason":s.failure_reason,"response_completed":s.response_completed,"duration_ms":s.process_time*1000.0,"first_output_ms":s.first_output_ms,"response_created_ms":s.response_created_ms,"first_text_ms":s.first_text_ms,"input_tokens":s.fact_usage.input,"output_tokens":s.fact_usage.output,"cache_read_tokens":s.fact_usage.cache_read,"cache_write_tokens":s.fact_usage.cache_write,"cache_write_1h_tokens":s.fact_usage.cache_write_1h})
}

pub fn attempt_event(s: &crate::storage::database::ChannelStat) -> Value {
    let at = now_ms();
    let attempt_id = if s.attempt_id.is_empty() {
        format!("{}-unknown", s.request_id)
    } else {
        s.attempt_id.clone()
    };
    json!({"schema":1,"kind":"attempt","event_id":new_event_id("attempt"),"instance_id":instance_id(),"at_ms":at,"request_id":s.request_id,"attempt_id":attempt_id,"transport_timing":s.transport_timing,"first_output_ms":s.first_output_ms,"response_created_ms":s.response_created_ms,"first_text_ms":s.first_text_ms,"duration_ms":s.duration_ms,"key_id":format!("key-{}",hex_sha(&s.api_key)),"provider":s.provider,"model":s.model,"upstream_model":s.upstream_model,"endpoint":s.endpoint,"stream":s.stream,"outcome":if s.success{"success"}else{"failed"},"terminal_kind":s.terminal_kind,"failure_reason":s.failure_reason,"response_completed":s.response_completed})
}

pub fn dispatch_event(
    key: &crate::observability::metrics::MetricKey,
    request_id: &str,
    attempt_id: &str,
    key_id: &str,
    elapsed_ms: f64,
) -> Value {
    json!({"schema":1,"kind":"dispatch","key_id":key_id,"event_id":new_event_id("dispatch"),"instance_id":instance_id(),"at_ms":now_ms(),"request_id":request_id,"attempt_id":attempt_id,"provider":key.provider,"model":key.model,"upstream_model":key.upstream_model,"endpoint":key.endpoint,"stream":key.stream,"dispatch_ms":elapsed_ms})
}

pub(crate) fn instance_id() -> String {
    static INSTANCE: OnceLock<String> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            std::env::var("INSTANCE_ID")
                .unwrap_or_else(|_| format!("uni-api-{}", std::process::id()))
        })
        .clone()
}

// Generate once when creating the fact. Retries replay the serialized value;
// caller request IDs may repeat and must never be storage uniqueness keys.
pub(crate) fn new_event_id(kind: &str) -> String {
    static PROCESS: OnceLock<String> = OnceLock::new();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let process = PROCESS.get_or_init(|| {
        let mut random = [0u8; 16];
        if SystemRandom::new().fill(&mut random).is_ok() {
            hex::encode(random)
        } else {
            hex_sha(&format!(
                "{:?}-{}-{}",
                SystemTime::now(),
                std::process::id(),
                std::env::var("HOSTNAME").unwrap_or_default()
            ))
        }
    });
    format!(
        "{kind}-{process}-{:016x}",
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::database::ChannelStat;
    use crate::storage::database::RequestStat;

    #[test]
    fn repeated_caller_ids_do_not_collide_and_replays_keep_the_created_fact() {
        let request = RequestStat {
            request_id: "req-1".into(),
            ..RequestStat::default()
        };
        let first = request_event(&request);
        let second = request_event(&request);
        assert_ne!(first["event_id"], second["event_id"]);
        let replay: Value = serde_json::from_str(&first.to_string()).unwrap();
        assert_eq!(first, replay);
        assert!(first["event_id"].as_str().unwrap().starts_with("request-"));

        let attempt = ChannelStat {
            request_id: "req-1".into(),
            attempt_id: "req-1-r2".into(),
            provider: "provider".into(),
            model: "model".into(),
            ..ChannelStat::default()
        };
        let event = attempt_event(&attempt);
        assert!(event["event_id"].as_str().unwrap().starts_with("attempt-"));
        assert_eq!(event["attempt_id"], "req-1-r2");
        assert_ne!(event["event_id"], attempt_event(&attempt)["event_id"]);
    }

    #[test]
    fn dispatch_carries_same_caller_fingerprint_as_attempt_without_secrets() {
        let token = "caller-secret";
        let id = crate::routing::catalog::key_id(token);
        let metric =
            crate::observability::metrics::MetricKey::new("p", "m", "m", "/v1/responses", true);
        let d = dispatch_event(&metric, "r", "r-1", &id, 25.0);
        let a = attempt_event(&ChannelStat {
            api_key: token.into(),
            request_id: "r".into(),
            attempt_id: "r-1".into(),
            ..Default::default()
        });
        assert_eq!(d["key_id"], a["key_id"]);
        assert_eq!(d["dispatch_ms"], 25.0);
        assert!(!d.to_string().contains(token));
    }
    #[test]
    fn request_event_preserves_cache_usage_as_nullable_facts() {
        let event = request_event(&RequestStat {
            request_id: "req-cache".into(),
            prompt_tokens: 100,
            completion_tokens: 20,
            fact_usage: crate::observability::usage::FactUsage {
                input: Some(100),
                output: Some(20),
                cache_read: Some(40),
                cache_write: Some(10),
                cache_write_1h: Some(3),
            },
            stream: true,
            status: 200,
            upstream_model: "upstream".into(),
            ..RequestStat::default()
        });
        assert_eq!(event["input_tokens"], 100);
        assert_eq!(event["cache_read_tokens"], 40);
        assert_eq!(event["cache_write_tokens"], 10);
        assert_eq!(event["cache_write_1h_tokens"], 3);
        assert_eq!(event["output_tokens"], 20);
        assert_eq!(event["stream"], true);
        assert_eq!(event["status"], 200);
        assert_eq!(event["upstream_model"], "upstream");
        let unknown = request_event(&RequestStat::default());
        assert!(unknown["input_tokens"].is_null());
        let zero = request_event(&RequestStat {
            fact_usage: crate::observability::usage::FactUsage {
                input: Some(0),
                output: Some(0),
                cache_read: Some(0),
                ..Default::default()
            },
            ..Default::default()
        });
        assert_eq!(zero["input_tokens"], 0);
        assert_eq!(zero["cache_read_tokens"], 0);
    }

    #[tokio::test]
    async fn batch_wait_coalesces_facts_that_arrive_after_the_first() {
        let (tx, mut rx) = mpsc::channel(16);
        tx.send(json!({"event_id":"first"})).await.unwrap();
        let send_later = async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            tx.send(json!({"event_id":"second"})).await.unwrap();
        };
        let (batch, ()) = tokio::join!(receive_batch(&mut rx, Duration::from_secs(1)), send_later);
        assert_eq!(batch.unwrap().len(), 2);
        assert!(receive_batch(&mut rx, Duration::from_secs(1))
            .await
            .is_none());
    }

    #[tokio::test]
    async fn closed_queue_drains_every_batch_in_order() {
        let count = BATCH_SIZE * 2 + 7;
        let (tx, mut rx) = mpsc::channel(count);
        for i in 0..count {
            tx.send(json!(i)).await.unwrap();
        }
        drop(tx);
        let mut actual = Vec::new();
        let mut sizes = Vec::new();
        while let Some(batch) = receive_batch(&mut rx, Duration::from_secs(1)).await {
            sizes.push(batch.len());
            actual.extend(batch);
        }
        assert_eq!(sizes, vec![BATCH_SIZE, BATCH_SIZE, 7]);
        assert_eq!(actual, (0..count).map(|i| json!(i)).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn sparse_batch_flushes_while_sender_remains_open() {
        let (tx, mut rx) = mpsc::channel(16);
        tx.send(json!(1)).await.unwrap();
        let batch = tokio::time::timeout(
            Duration::from_secs(1),
            receive_batch(&mut rx, Duration::from_millis(20)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(batch, vec![json!(1)]);
        assert!(!tx.is_closed());
    }

    #[test]
    fn batch_object_key_is_stable_for_retries() {
        let config = UploadConfig {
            client: Client::new(),
            endpoint: "https://example.invalid".into(),
            token: "fixture".into(),
            instance: "instance".into(),
            epoch: "fixture-epoch".into(),
            spool: PathBuf::from("/tmp/facts-test"),
        };
        let batch = vec![json!({"event_id":"one","schema":1})];
        let (key_a, body_a) = batch_payload(&config, &batch).expect("payload");
        let (key_b, body_b) = batch_payload(&config, &batch).expect("payload");
        assert_eq!(body_a, body_b);
        assert_eq!(key_a, key_b);
        assert!(key_a.contains(&hex_sha(&body_a)));
    }

    #[test]
    fn different_batches_have_different_object_keys() {
        let config = UploadConfig {
            client: Client::new(),
            endpoint: "https://example.invalid".into(),
            token: "fixture".into(),
            instance: "instance".into(),
            epoch: "fixture-epoch".into(),
            spool: PathBuf::from("/tmp/facts-test"),
        };
        let a = batch_payload(&config, &[json!({"event_id":"one"})])
            .unwrap()
            .0;
        let b = batch_payload(&config, &[json!({"event_id":"two"})])
            .unwrap()
            .0;
        assert_ne!(a, b);
    }
}
#[test]
fn ingest_target_preserves_source_and_public_proxy_prefix() {
    assert_eq!(
        ingest_url("https://console.test/analytics/v1/runtime-restore/primary").as_deref(),
        Some("https://console.test/analytics/v1/runtime-facts/primary")
    );
    assert!(ingest_url("https://console.test/bootstrap").is_none());
    assert!(ingest_url("https://console.test/v1/runtime-restore/one/two").is_none());
}

#[tokio::test]
async fn ingest_only_acknowledges_matching_digest_and_never_follows_redirects() {
    use axum::{
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
        Json, Router,
    };
    use std::sync::Arc;
    let mode = Arc::new(AtomicU64::new(0));
    async fn handler(
        State(mode): State<Arc<AtomicU64>>,
        headers: HeaderMap,
        body: String,
    ) -> (StatusCode, HeaderMap, Json<Value>) {
        assert_eq!(
            headers.get("authorization").unwrap(),
            "Bearer fixture-secret"
        );
        let mut response_headers = HeaderMap::new();
        let value = mode.load(Ordering::Relaxed);
        if value == 2 {
            response_headers.insert(
                "location",
                "http://127.0.0.1:1/credential-trap".parse().unwrap(),
            );
        }
        (
            if value == 2 {
                StatusCode::TEMPORARY_REDIRECT
            } else {
                StatusCode::OK
            },
            response_headers,
            Json(
                json!({"format":"facts-ack-v1","sha256":if value==0{hex_sha(&body)}else{"wrong".into()}}),
            ),
        )
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/facts", post(handler))
        .with_state(mode.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let c = UploadConfig {
        client: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
        endpoint: format!("http://{address}/facts"),
        token: "fixture-secret".into(),
        instance: "fixture".into(),
        epoch: "fixture-epoch".into(),
        spool: std::env::temp_dir().join("absent-facts-spool"),
    };
    assert!(upload_body(&c, "", "{}\n").await.is_ok());
    mode.store(1, Ordering::Relaxed);
    assert_eq!(
        upload_body(&c, "", "{}\n").await.unwrap_err(),
        "fact acknowledgement mismatch"
    );
    mode.store(2, Ordering::Relaxed);
    assert_eq!(
        upload_body(&c, "", "{}\n").await.unwrap_err(),
        "fact ingestion HTTP 307"
    );
    server.abort();
}

#[tokio::test]
async fn spool_capacity_evicts_oldest_file_before_reserving_new_bytes() {
    let root = std::env::temp_dir().join(new_event_id("spool-test"));
    std::fs::create_dir(&root).unwrap();
    let c = UploadConfig {
        client: Client::new(),
        endpoint: "http://127.0.0.1/unused".into(),
        token: "fixture".into(),
        instance: "fixture".into(),
        epoch: "epoch".into(),
        spool: root.clone(),
    };
    for name in ["old", "middle", "new"] {
        std::fs::File::create(root.join(format!("{name}.jsonl")))
            .unwrap()
            .set_len(30 * 1024 * 1024)
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(reserve_spool(&c, 4 * 1024 * 1024).await);
    assert!(!root.join("old.jsonl").exists());
    assert!(root.join("middle.jsonl").exists());
    assert!(root.join("new.jsonl").exists());
    std::fs::remove_dir_all(root).unwrap();
}
