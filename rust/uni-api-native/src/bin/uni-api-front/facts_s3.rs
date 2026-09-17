//! Durable immutable request-fact exporter. The bounded queue is outside the
//! request critical path: full or unavailable storage drops facts with an
//! explicit counter, while serving remains available.
use hmac::{Hmac, Mac};
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
type HmacSha256 = Hmac<Sha256>;
const QUEUE_CAPACITY: usize = 4096;
const BATCH_SIZE: usize = 128;
const BATCH_WAIT: Duration = Duration::from_secs(2);
const UPLOAD_RETRIES: usize = 3;

#[derive(Clone)]
struct UploadConfig {
    client: Client,
    endpoint: String,
    bucket: String,
    prefix: String,
    access: String,
    secret: String,
    token: Option<String>,
    instance: String,
    spool: PathBuf,
}
#[derive(Clone)]
pub struct FactWriter {
    sender: mpsc::Sender<Value>,
}
impl FactWriter {
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("FACTS_S3_ENDPOINT")
            .ok()?
            .trim_end_matches('/')
            .to_string();
        let bucket = std::env::var("FACTS_S3_BUCKET").ok()?.trim().to_string();
        let access = std::env::var("FACTS_S3_ACCESS_KEY_ID")
            .or_else(|_| std::env::var("AWS_ACCESS_KEY_ID"))
            .ok()?
            .trim()
            .to_string();
        let secret = std::env::var("FACTS_S3_SECRET_ACCESS_KEY")
            .or_else(|_| std::env::var("AWS_SECRET_ACCESS_KEY"))
            .ok()?
            .trim()
            .to_string();
        if Url::parse(&endpoint).ok()?.host_str().is_none() || bucket.is_empty() {
            return None;
        }
        let (sender, mut rx) = mpsc::channel(QUEUE_CAPACITY);
        let config = UploadConfig {
            client: Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .ok()?,
            endpoint,
            bucket,
            prefix: std::env::var("FACTS_S3_PREFIX").unwrap_or_else(|_| "uni-api-facts/v1".into()),
            access,
            secret,
            token: std::env::var("FACTS_S3_SESSION_TOKEN").ok(),
            instance: std::env::var("INSTANCE_ID")
                .unwrap_or_else(|_| format!("uni-api-{}", std::process::id())),
            spool: PathBuf::from(
                std::env::var("FACTS_S3_SPOOL_DIR").unwrap_or_else(|_| "./data/facts-spool".into()),
            ),
        };
        if std::fs::create_dir_all(&config.spool).is_err() {
            eprintln!("facts_s3_init_failed reason=spool_unavailable");
            return None;
        }
        let retry_config = config.clone();
        tokio::spawn(async move {
            retry_spool(retry_config).await;
        });
        tokio::spawn(async move {
            while let Some(batch) = receive_batch(&mut rx, BATCH_WAIT).await {
                let Some((key, body)) = batch_payload(&config, &batch) else {
                    continue;
                };
                let spool = persist_batch(&config, &key, &body).await;
                if let Err(error) = upload_body_with_retry(&config, &key, &body).await {
                    eprintln!(
                        "{{\"event_type\":\"facts_s3_upload_error\",\"error\":{:?},\"spooled\":{}}}",
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
        if self.sender.try_send(event).is_err() {
            eprintln!("facts_s3_queue_dropped record_type=request_fact");
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

fn batch_payload(c: &UploadConfig, batch: &[Value]) -> Option<(String, String)> {
    let body = batch
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .ok()?
        .join("\n")
        + "\n";
    let digest = hex_sha(&body);
    let key = format!(
        "{}/{}/{}/batch-{}.jsonl",
        c.prefix.trim_matches('/'),
        c.instance,
        "immutable",
        digest
    );
    Some((key, body))
}

async fn upload_body(c: &UploadConfig, key: &str, body: &str) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?;
    let secs = now.as_secs();
    let date = chrono_date(secs);
    let amz = chrono_timestamp(secs);
    let hash = hex_sha(body);
    let url = format!("{}/{}/{}", c.endpoint, c.bucket, key);
    let parsed = Url::parse(&url).map_err(|e| e.to_string())?;
    let host = parsed.host_str().ok_or("S3 endpoint host missing")?;
    let headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, hash, amz
    );
    let signed = "host;x-amz-content-sha256;x-amz-date";
    let canonical = format!(
        "PUT\n/{}/{}\n\n{}\n{}\n{}",
        c.bucket, key, headers, signed, hash
    );
    let scope = format!("{}/auto/s3/aws4_request", date);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz,
        scope,
        hex_sha(&canonical)
    );
    let kdate = hmac(format!("AWS4{}", c.secret).as_bytes(), date.as_bytes());
    let kregion = hmac(&kdate, b"auto");
    let kservice = hmac(&kregion, b"s3");
    let signing = hmac(&kservice, b"aws4_request");
    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        c.access,
        scope,
        signed,
        hex::encode(hmac(&signing, string_to_sign.as_bytes()))
    );
    let mut req = c
        .client
        .put(url)
        .header("Host", host)
        .header("x-amz-content-sha256", hash)
        .header("x-amz-date", amz)
        .header("Authorization", auth)
        .header("Content-Type", "application/x-ndjson")
        .body(body.to_owned());
    if let Some(token) = &c.token {
        req = req.header("x-amz-security-token", token)
    }
    let resp = req
        .send()
        .await
        .map_err(|_| "S3 fact upload failed".to_string())?;
    if !resp.status().is_success() {
        return Err(format!("S3 fact upload HTTP {}", resp.status()));
    }
    Ok(())
}

async fn persist_batch(c: &UploadConfig, key: &str, body: &str) -> bool {
    let digest = hex_sha(body);
    let target = c.spool.join(format!("{}.jsonl", digest));
    if tokio::fs::try_exists(&target).await.unwrap_or(false) {
        return true;
    }
    let pending = c.spool.join(format!("{}.pending", digest));
    if tokio::fs::write(&pending, body.as_bytes()).await.is_err() {
        eprintln!("facts_s3_spool_write_failed digest={digest}");
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
        eprintln!("facts_s3_spool_commit_failed digest={digest}");
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
                let key = format!(
                    "{}/{}/{}/batch-{}.jsonl",
                    c.prefix.trim_matches('/'),
                    c.instance,
                    "immutable",
                    name
                );
                if upload_body_with_retry(&c, &key, &body).await.is_ok() {
                    let _ = tokio::fs::remove_file(path).await;
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

async fn upload_body_with_retry(c: &UploadConfig, key: &str, body: &str) -> Result<(), String> {
    let mut last = String::from("S3 fact upload failed");
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
fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC key");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}
fn hex_sha(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}
fn chrono_date(secs: u64) -> String {
    let days = secs / 86400;
    let z = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    format!("{:04}{:02}{:02}", year, m, d)
}
fn chrono_timestamp(secs: u64) -> String {
    format!(
        "{}T{:02}{:02}{:02}Z",
        chrono_date(secs),
        ((secs % 86400) / 3600),
        ((secs % 3600) / 60),
        secs % 60
    )
}
static WRITER: OnceLock<Option<FactWriter>> = OnceLock::new();
pub fn global() -> Option<FactWriter> {
    WRITER.get_or_init(FactWriter::from_env).clone()
}
pub fn request_event(s: &crate::persistence::RequestStat) -> Value {
    let key = hex_sha(&s.api_key);
    let at = now_ms();
    json!({"schema":1,"kind":"request","event_id":new_event_id("request"),"instance_id":instance_id(),"at_ms":at,"request_id":s.request_id,"trace_id":s.trace_id,"key_id":format!("key-{}",key),"endpoint":s.endpoint,"provider":s.provider,"model":s.model,"upstream_model":s.upstream_model,"stream":s.stream,"status":s.status,"outcome":if s.is_flagged{"failed"}else{"success"},"duration_ms":s.process_time*1000.0,"first_output_ms":s.first_output_ms,"input_tokens":s.fact_usage.input,"output_tokens":s.fact_usage.output,"cache_read_tokens":s.fact_usage.cache_read,"cache_write_tokens":s.fact_usage.cache_write,"cache_write_1h_tokens":s.fact_usage.cache_write_1h})
}
pub fn attempt_event(s: &crate::persistence::ChannelStat) -> Value {
    let at = now_ms();
    let attempt_id = if s.attempt_id.is_empty() {
        format!("{}-unknown", s.request_id)
    } else {
        s.attempt_id.clone()
    };
    json!({"schema":1,"kind":"attempt","event_id":new_event_id("attempt"),"instance_id":instance_id(),"at_ms":at,"request_id":s.request_id,"attempt_id":attempt_id,"first_output_ms":s.first_output_ms,"duration_ms":s.duration_ms,"key_id":format!("key-{}",hex_sha(&s.api_key)),"provider":s.provider,"model":s.model,"upstream_model":s.upstream_model,"endpoint":s.endpoint,"stream":s.stream,"outcome":if s.success{"success"}else{"failed"}})
}

pub fn dispatch_event(
    key: &crate::channel_metrics::MetricKey,
    request_id: &str,
    attempt_id: &str,
    key_id: &str,
    elapsed_ms: f64,
) -> Value {
    json!({"schema":1,"kind":"dispatch","key_id":key_id,"event_id":new_event_id("dispatch"),"instance_id":instance_id(),"at_ms":now_ms(),"request_id":request_id,"attempt_id":attempt_id,"provider":key.provider,"model":key.model,"upstream_model":key.upstream_model,"endpoint":key.endpoint,"stream":key.stream,"dispatch_ms":elapsed_ms})
}

fn instance_id() -> String {
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
fn new_event_id(kind: &str) -> String {
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
    use crate::persistence::{ChannelStat, RequestStat};

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
        let id = crate::channel_catalog::key_id(token);
        let metric = crate::channel_metrics::MetricKey::new("p", "m", "m", "/v1/responses", true);
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
            fact_usage: crate::fact_usage::FactUsage {
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
            fact_usage: crate::fact_usage::FactUsage {
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
            bucket: "bucket".into(),
            prefix: "facts/v1".into(),
            access: "access".into(),
            secret: "secret".into(),
            token: None,
            instance: "instance".into(),
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
            bucket: "bucket".into(),
            prefix: "facts/v1".into(),
            access: "access".into(),
            secret: "secret".into(),
            token: None,
            instance: "instance".into(),
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
