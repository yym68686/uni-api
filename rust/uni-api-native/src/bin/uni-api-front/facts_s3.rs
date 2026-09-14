//! Durable immutable request-fact exporter. The bounded queue is outside the
//! request critical path: full or unavailable storage drops facts with an
//! explicit counter, while serving remains available.
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
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
            client: Client::new(),
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
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            loop {
                let first = tokio::select! {v=rx.recv()=>v,_=tokio::time::sleep(BATCH_WAIT)=>{if batch.is_empty(){continue}else{None}}};
                if let Some(v) = first {
                    batch.push(v)
                } else if batch.is_empty() {
                    if rx.is_closed() {
                        break;
                    } else {
                        continue;
                    }
                };
                while batch.len() < BATCH_SIZE {
                    match rx.try_recv() {
                        Ok(v) => batch.push(v),
                        Err(_) => break,
                    }
                }
                if let Err(error) = upload_batch_with_retry(&config, &batch).await {
                    persist_failed_batch(&config, &batch).await;
                    eprintln!(
                        "{{\"event_type\":\"facts_s3_upload_error\",\"error\":{:?}}}",
                        error
                    )
                };
                batch.clear();
                if rx.is_closed() {
                    break;
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
async fn upload_batch(c: &UploadConfig, batch: &[Value]) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?;
    let secs = now.as_secs();
    let date = chrono_date(secs);
    let amz = chrono_timestamp(secs);
    let body = batch
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?
        .join("\n")
        + "\n";
    let hash = hex_sha(&body);
    // Content-addressed object names make retries idempotent even if a response
    // is lost after R2 has committed the object.
    let nonce = hex_sha(&body);
    let key = format!(
        "{}/{}/{}/batch-{}-{}.jsonl",
        c.prefix.trim_matches('/'),
        secs / 86400,
        c.instance,
        secs,
        nonce
    );
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
        .body(body);
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

async fn persist_failed_batch(c: &UploadConfig, batch: &[Value]) {
    let body = match batch
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(lines) => lines.join("\n") + "\n",
        Err(_) => return,
    };
    let name = format!("{}.jsonl", hex_sha(&body));
    let pending = c.spool.join(format!("{}.pending", name));
    let target = c.spool.join(name);
    if tokio::fs::write(&pending, body.as_bytes()).await.is_ok() {
        let _ = tokio::fs::rename(pending, target).await;
    }
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
                    now_ms() as u64 / 1000 / 86400,
                    c.instance,
                    name
                );
                if upload_body(&c, &key, &body).await.is_ok() {
                    let _ = tokio::fs::remove_file(path).await;
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

async fn upload_body(c: &UploadConfig, _key: &str, body: &str) -> Result<(), String> {
    let batch: Vec<Value> = body
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if batch.is_empty() {
        return Ok(());
    }
    upload_batch(c, &batch).await
}

async fn upload_batch_with_retry(c: &UploadConfig, batch: &[Value]) -> Result<(), String> {
    let mut last = String::from("S3 fact upload failed");
    for attempt in 0..UPLOAD_RETRIES {
        match upload_batch(c, batch).await {
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
    let stream = s.timing_spans.contains("\\\"stream\\\":true")
        || s.timing_spans.contains("\\\"streaming\\\":true");
    let at = now_ms();
    json!({"schema":1,"kind":"request","event_id":format!("request-{}-{}",s.request_id,at),"at_ms":at,"request_id":s.request_id,"trace_id":s.trace_id,"key_id":format!("key-{}",key),"endpoint":s.endpoint,"provider":s.provider,"model":s.model,"upstream_model":s.model,"stream":stream,"outcome":if s.is_flagged{"failed"}else{"success"},"duration_ms":s.process_time*1000.0,"first_output_ms":(s.first_response_time>0.0).then_some(s.first_response_time*1000.0),"input_tokens":s.prompt_tokens,"output_tokens":s.completion_tokens})
}
pub fn attempt_event(s: &crate::persistence::ChannelStat) -> Value {
    let at = now_ms();
    json!({"schema":1,"kind":"attempt","event_id":format!("attempt-{}-{}-{}-{}",s.request_id,s.provider,s.model,at),"at_ms":at,"request_id":s.request_id,"provider":s.provider,"model":s.model,"upstream_model":s.model,"stream":false,"outcome":if s.success{"success"}else{"failed"}})
}
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}
