use std::collections::{HashMap, HashSet};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::idempotency;
use crate::proxy::{filtered_response_headers, json_error, AppState};

const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";
const MAX_ERROR_BODY_BYTES: usize = 1024 * 1024;
const DOWNSTREAM_SEGMENT_BYTES: usize = 64 * 1024;
const DOWNSTREAM_CHANNEL_SEGMENTS: usize = 16;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[derive(Clone, Debug, Deserialize)]
struct Plan {
    attempt_id: String,
    url: String,
    headers: HashMap<String, String>,
    body: String,
    proxy: Option<String>,
    engine: String,
    #[serde(default)]
    http1_only: bool,
    #[serde(default = "default_commit_policy")]
    commit_policy: String,
    #[serde(default)]
    normalize_custom_tool_call_ids: bool,
    first_byte_timeout_seconds: Option<f64>,
    idle_timeout_seconds: Option<f64>,
    total_timeout_seconds: Option<f64>,
    #[serde(default = "default_max_event_bytes")]
    max_event_bytes: usize,
    #[serde(default = "default_max_precommit_items")]
    max_precommit_items: usize,
    #[serde(default = "default_max_precommit_bytes")]
    max_precommit_bytes: usize,
}

fn default_commit_policy() -> String {
    "real_output".to_owned()
}

fn default_max_event_bytes() -> usize {
    8 * 1024 * 1024
}

fn default_max_precommit_items() -> usize {
    128
}

fn default_max_precommit_bytes() -> usize {
    8 * 1024 * 1024 + 128 * 266
}

#[derive(Default)]
struct StreamStats {
    upstream_bytes: u64,
    downstream_bytes: u64,
    event_count: u64,
    delta_events: u64,
    normalized_events: u64,
    usage: Option<Value>,
    wire_hash: Sha256,
}

impl StreamStats {
    fn report(&self) -> Value {
        let hash = self.wire_hash.clone().finalize();
        json!({
            "upstream_bytes": self.upstream_bytes,
            "downstream_bytes": self.downstream_bytes,
            "event_count": self.event_count,
            "delta_events": self.delta_events,
            "normalized_events": self.normalized_events,
            "usage": self.usage,
            "wire_sha256": format!("{hash:x}"),
        })
    }

    fn observe_wire(&mut self, wire: &[u8]) {
        self.downstream_bytes = self.downstream_bytes.saturating_add(wire.len() as u64);
        self.wire_hash.update(wire);
    }
}

#[derive(Debug)]
enum Terminal {
    Completed,
    Incomplete,
    SemanticFailure { event_type: String, payload: Value },
}

struct ActiveAttempt {
    plan: Plan,
    status: StatusCode,
    headers: HeaderMap,
    stream: ByteStream,
    decoder: SseDecoder,
    processor: ResponsesProcessor,
    early_output: Vec<Bytes>,
    buffered: Vec<Bytes>,
    stats: StreamStats,
    terminal: Option<Terminal>,
    total_deadline: Option<tokio::time::Instant>,
    commit_reason: &'static str,
    precommit_keepalive_sent: bool,
    business_committed: bool,
}

enum PreflightResult {
    Retry(Value),
    Started(ActiveAttempt),
}

pub async fn serve_session(
    state: AppState,
    session_id: String,
    control_response: reqwest::Response,
    mut idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let public_control_headers = public_control_headers(control_response.headers());
    let mut control_response = Some(control_response);

    let mut message = match control_get_plan(&state, &session_id).await {
        Ok(message) => message,
        Err(error) => {
            release_owner(&mut idempotency_owner).await;
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Rust Responses control plan failed: {error}"),
            );
        }
    };

    loop {
        if message.get("kind").and_then(Value::as_str) == Some("final") {
            return response_from_final(
                &message,
                &public_control_headers,
                idempotency_owner.take(),
            )
            .await;
        }
        let plan = match serde_json::from_value::<Plan>(message.clone()) {
            Ok(plan) if message.get("kind").and_then(Value::as_str) == Some("plan") => plan,
            Ok(_) | Err(_) => {
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_GATEWAY, "Invalid Rust Responses plan");
            }
        };
        match preflight_attempt(&state, plan, false).await {
            Ok(PreflightResult::Retry(mut outcome)) => {
                outcome["attempt_id"] = Value::String(
                    message
                        .get("attempt_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                );
                message = match control_advance(&state, &session_id, &outcome).await {
                    Ok(next) => next,
                    Err(error) => {
                        release_owner(&mut idempotency_owner).await;
                        return json_error(
                            StatusCode::BAD_GATEWAY,
                            &format!("Rust Responses retry coordination failed: {error}"),
                        );
                    }
                };
            }
            Ok(PreflightResult::Started(active)) => {
                let observation = json!({
                    "attempt_id": active.plan.attempt_id,
                    "upstream_status_code": active.status.as_u16(),
                    "commit_reason": active.commit_reason,
                    "business_committed": active.business_committed,
                    "precommit_events": active.stats.event_count,
                    "precommit_bytes": active.buffered.iter().map(|item| item.len() as u64).sum::<u64>(),
                });
                if let Err(error) = control_commit(&state, &session_id, &observation).await {
                    release_owner(&mut idempotency_owner).await;
                    return json_error(
                        StatusCode::BAD_GATEWAY,
                        &format!("Rust Responses commit coordination failed: {error}"),
                    );
                }
                let control_drain = control_response.take().map(|response| {
                    tokio::spawn(async move {
                        let mut stream = response.bytes_stream();
                        while let Some(result) = stream.next().await {
                            if result.is_err() {
                                break;
                            }
                        }
                    })
                });
                return start_public_stream(
                    state,
                    session_id,
                    active,
                    public_control_headers,
                    control_drain,
                    idempotency_owner.take(),
                );
            }
            Err(error) => {
                let outcome = json!({
                    "attempt_id": message.get("attempt_id").and_then(Value::as_str).unwrap_or_default(),
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
                message = match control_advance(&state, &session_id, &outcome).await {
                    Ok(next) => next,
                    Err(control_error) => {
                        release_owner(&mut idempotency_owner).await;
                        return json_error(
                            StatusCode::BAD_GATEWAY,
                            &format!("Rust Responses preflight failed: {control_error}"),
                        );
                    }
                };
            }
        }
    }
}

async fn preflight_attempt(
    state: &AppState,
    plan: Plan,
    keepalive_already_sent: bool,
) -> Result<PreflightResult, String> {
    let client = state
        .upstream_client(plan.proxy.as_deref(), plan.http1_only)
        .await?;
    let mut headers = HeaderMap::new();
    for (name, value) in &plan.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            return Err("upstream plan contains an invalid header name".into());
        };
        let Ok(value) = HeaderValue::from_str(value) else {
            return Err("upstream plan contains an invalid header value".into());
        };
        headers.append(name, value);
    }
    headers
        .entry("accept-encoding")
        .or_insert(HeaderValue::from_static("identity"));

    let started_at = tokio::time::Instant::now();
    let first_deadline = deadline(started_at, plan.first_byte_timeout_seconds);
    let total_deadline = deadline(started_at, plan.total_timeout_seconds);
    let send_deadline = earlier_deadline(first_deadline, total_deadline);
    let request = client
        .post(&plan.url)
        .headers(headers)
        .body(plan.body.clone());
    let response = await_deadline(request.send(), send_deadline)
        .await
        .map_err(|error| format!("upstream response headers failed: {error}"))?
        .map_err(|error| format!("upstream response headers failed: {error}"))?;
    let status = response.status();
    let unsupported_encoding = response
        .headers()
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"));
    let response_headers = filtered_response_headers(response.headers());
    if !status.is_success() {
        let body = read_limited_body(response, total_deadline).await;
        return Ok(PreflightResult::Retry(json!({
            "kind": "http_error",
            "status_code": status.as_u16(),
            "upstream_status_code": status.as_u16(),
            "body": body,
            "committed": false,
        })));
    }
    if unsupported_encoding {
        return Ok(PreflightResult::Retry(json!({
            "kind": "protocol_error",
            "status_code": 502,
            "upstream_status_code": status.as_u16(),
            "detail": "Responses upstream ignored Accept-Encoding: identity",
            "committed": false,
        })));
    }

    let stream = Box::pin(response.bytes_stream());
    let mut active = ActiveAttempt {
        decoder: SseDecoder::new(plan.max_event_bytes),
        processor: ResponsesProcessor::new(
            plan.commit_policy.clone(),
            plan.normalize_custom_tool_call_ids,
        ),
        plan,
        status,
        headers: response_headers,
        stream,
        early_output: Vec::new(),
        buffered: Vec::new(),
        stats: StreamStats::default(),
        terminal: None,
        total_deadline,
        commit_reason: "real_output",
        precommit_keepalive_sent: keepalive_already_sent,
        business_committed: false,
    };

    loop {
        let next_deadline = earlier_deadline(first_deadline, active.total_deadline);
        let chunk = match await_deadline(active.stream.next(), next_deadline).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(error))) => {
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 502,
                    "upstream_status_code": status.as_u16(),
                    "detail": format!("upstream stream read failed: {error}"),
                    "committed": false,
                })))
            }
            Ok(None) => {
                let frames = active.decoder.finish()?;
                if !frames.is_empty() {
                    if let Some(result) = process_preflight_frames(&mut active, frames)? {
                        return Ok(result);
                    }
                }
                return Ok(PreflightResult::Retry(json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "upstream_status_code": status.as_u16(),
                    "detail": "Responses upstream closed before substantive output",
                    "committed": false,
                })));
            }
            Err(error) => {
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 504,
                    "upstream_status_code": status.as_u16(),
                    "detail": error,
                    "committed": false,
                })))
            }
        };
        active.stats.upstream_bytes = active
            .stats
            .upstream_bytes
            .saturating_add(chunk.len() as u64);
        let frames = active.decoder.feed(&chunk)?;
        if let Some(result) = process_preflight_frames(&mut active, frames)? {
            return Ok(result);
        }
    }
}

fn process_preflight_frames(
    active: &mut ActiveAttempt,
    frames: Vec<SseFrame>,
) -> Result<Option<PreflightResult>, String> {
    let mut started = false;
    for frame in frames {
        let processed = active.processor.process(frame, &mut active.stats)?;
        if let Some(Terminal::SemanticFailure {
            event_type,
            payload,
        }) = processed.terminal.as_ref()
        {
            if !started {
                return Ok(Some(PreflightResult::Retry(json!({
                    "kind": "semantic_error",
                    "status_code": 502,
                    "upstream_status_code": active.status.as_u16(),
                    "event_type": event_type,
                    "payload": payload,
                    "committed": false,
                }))));
            }
            active.terminal = processed.terminal;
            return Ok(Some(PreflightResult::Started(take_active(active)?)));
        }
        let early_keepalive = !active.precommit_keepalive_sent
            && processed.canonical_keepalive
            && active.plan.engine == "codex";
        let suppress_repeated_keepalive =
            active.precommit_keepalive_sent && processed.event_type.as_deref() == Some("keepalive");
        if processed.event_type.as_deref() == Some("response.created")
            && !active.precommit_keepalive_sent
        {
            active.buffered.push(Bytes::from_static(
                b"event: keepalive\ndata: {\"type\":\"keepalive\",\"sequence_number\":0}\n\n",
            ));
            active.precommit_keepalive_sent = true;
        }
        if let Some(wire) = processed.wire.filter(|_| !suppress_repeated_keepalive) {
            let buffered_bytes: usize = active.buffered.iter().map(Bytes::len).sum();
            if !started
                && (active.buffered.len() >= active.plan.max_precommit_items
                    || buffered_bytes.saturating_add(wire.len()) > active.plan.max_precommit_bytes)
            {
                return Ok(Some(PreflightResult::Retry(json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "upstream_status_code": active.status.as_u16(),
                    "detail": "Responses upstream precommit buffer limit exceeded",
                    "committed": false,
                }))));
            }
            if early_keepalive {
                active.early_output.push(wire);
            } else {
                active.buffered.push(wire);
            }
        }
        if processed.event_type.as_deref() == Some("keepalive") {
            active.precommit_keepalive_sent = true;
        }
        if let Some(terminal) = processed.terminal {
            active.terminal = Some(terminal);
            active.commit_reason = "semantic_terminal";
            active.business_committed = true;
            return Ok(Some(PreflightResult::Started(take_active(active)?)));
        }
        if early_keepalive {
            if !started {
                active.commit_reason = "upstream_keepalive";
            }
            started = true;
        }
        if processed.commits {
            if !started {
                active.commit_reason = "real_output";
            }
            started = true;
            active.business_committed = true;
        }
    }
    if started {
        Ok(Some(PreflightResult::Started(take_active(active)?)))
    } else {
        Ok(None)
    }
}

fn take_active(active: &mut ActiveAttempt) -> Result<ActiveAttempt, String> {
    let placeholder = ActiveAttempt {
        plan: active.plan.clone(),
        status: active.status,
        headers: HeaderMap::new(),
        stream: Box::pin(futures_util::stream::empty()),
        decoder: SseDecoder::new(active.plan.max_event_bytes),
        processor: ResponsesProcessor::new("real_output".into(), false),
        early_output: Vec::new(),
        buffered: Vec::new(),
        stats: StreamStats::default(),
        terminal: None,
        total_deadline: active.total_deadline,
        commit_reason: active.commit_reason,
        precommit_keepalive_sent: active.precommit_keepalive_sent,
        business_committed: active.business_committed,
    };
    Ok(std::mem::replace(active, placeholder))
}

fn start_public_stream(
    state: AppState,
    session_id: String,
    active: ActiveAttempt,
    control_headers: HeaderMap,
    control_drain: Option<tokio::task::JoinHandle<()>>,
    idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let status = active.status;
    let mut headers = active.headers.clone();
    for (name, value) in &control_headers {
        headers.insert(name.clone(), value.clone());
    }
    headers
        .entry("content-type")
        .or_insert(HeaderValue::from_static("text/event-stream"));
    headers.insert("x-uni-api-data-plane", HeaderValue::from_static("rust-v1"));

    let capture_headers = headers.clone();
    let capture = idempotency_owner.map(|owner| IdempotencyCapture {
        max_bytes: owner.max_response_bytes(),
        owner,
        status,
        headers: capture_headers,
        chunks: Vec::new(),
        bytes: 0,
        overflowed: false,
    });
    if capture.is_some() {
        idempotency::executed_header(&mut headers);
    }

    let (sender, receiver) = mpsc::channel(DOWNSTREAM_CHANNEL_SEGMENTS);
    let cancellation = CancellationToken::new();
    let finished = Arc::new(AtomicBool::new(false));
    let body_stream = GuardedBodyStream {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation.clone(),
        finished: finished.clone(),
    };
    let runtime = ActiveRuntime {
        output: OutputSink::new(sender, cancellation.clone(), capture),
        finished,
        control_drain,
    };
    tokio::spawn(run_active_attempt(state, session_id, active, runtime));

    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

struct IdempotencyCapture {
    owner: idempotency::Owner,
    status: StatusCode,
    headers: HeaderMap,
    chunks: Vec<Bytes>,
    bytes: usize,
    max_bytes: usize,
    overflowed: bool,
}

struct OutputSink {
    sender: mpsc::Sender<Result<Bytes, io::Error>>,
    cancellation: CancellationToken,
    downstream_open: bool,
    capture: Option<IdempotencyCapture>,
}

struct ActiveRuntime {
    output: OutputSink,
    finished: Arc<AtomicBool>,
    control_drain: Option<tokio::task::JoinHandle<()>>,
}

impl OutputSink {
    fn new(
        sender: mpsc::Sender<Result<Bytes, io::Error>>,
        cancellation: CancellationToken,
        capture: Option<IdempotencyCapture>,
    ) -> Self {
        Self {
            sender,
            cancellation,
            downstream_open: true,
            capture,
        }
    }

    async fn send_wire(&mut self, wire: Bytes) -> Result<(), ()> {
        for offset in (0..wire.len()).step_by(DOWNSTREAM_SEGMENT_BYTES) {
            let end = (offset + DOWNSTREAM_SEGMENT_BYTES).min(wire.len());
            let segment = wire.slice(offset..end);
            if let Some(capture) = self.capture.as_mut() {
                if !capture.overflowed {
                    let next_bytes = capture.bytes.saturating_add(segment.len());
                    if next_bytes > capture.max_bytes
                        || !capture.owner.try_reserve_inflight_response(segment.len())
                    {
                        capture.owner.release_inflight_response(capture.bytes);
                        capture.overflowed = true;
                        capture.chunks.clear();
                        capture.bytes = 0;
                    } else {
                        capture.bytes = next_bytes;
                        capture.chunks.push(segment.clone());
                    }
                }
            }
            if !self.downstream_open {
                continue;
            }
            let result = tokio::select! {
                _ = self.cancellation.cancelled() => None,
                result = tokio::time::timeout(
                    downstream_write_timeout(),
                    self.sender.send(Ok(segment)),
                ) => Some(result),
            };
            if !matches!(result, Some(Ok(Ok(())))) {
                if self.capture.is_some() {
                    self.downstream_open = false;
                    continue;
                }
                return Err(());
            }
        }
        Ok(())
    }

    async fn finish(mut self, cacheable: bool) {
        let Some(capture) = self.capture.take() else {
            return;
        };
        capture.owner.release_inflight_response(capture.bytes);
        if !cacheable {
            capture.owner.release().await;
        } else if capture.overflowed {
            capture.owner.nonreplayable("response_too_large").await;
        } else {
            capture
                .owner
                .complete(
                    capture.status,
                    capture.headers,
                    capture.chunks,
                    capture.bytes,
                )
                .await;
        }
    }
}

async fn run_active_attempt(
    state: AppState,
    session_id: String,
    mut active: ActiveAttempt,
    runtime: ActiveRuntime,
) {
    let ActiveRuntime {
        mut output,
        finished,
        mut control_drain,
    } = runtime;
    let cancellation = output.cancellation.clone();
    let cacheable = 'request: loop {
        if flush_initial_output(&mut active, &mut output)
            .await
            .is_err()
        {
            complete_disconnect(
                &state,
                &session_id,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
            )
            .await;
            break 'request false;
        }

        if let Some(terminal) = active.terminal.take() {
            if !active.business_committed {
                let outcome = semantic_retry_outcome(&active, terminal);
                match retry_after_public_start(&state, &session_id, outcome).await {
                    Ok(RetryResolution::Active(next)) => {
                        active = next;
                        continue 'request;
                    }
                    Ok(RetryResolution::Final(message)) => {
                        break 'request send_final_inband(&mut output, &message).await.is_ok();
                    }
                    Err(_) => break 'request false,
                }
            }
            finish_terminal(
                &state,
                &session_id,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
                terminal,
            )
            .await;
            break 'request true;
        }

        loop {
            let idle_deadline = deadline(
                tokio::time::Instant::now(),
                active.plan.idle_timeout_seconds,
            );
            let next_deadline = earlier_deadline(idle_deadline, active.total_deadline);
            let next = tokio::select! {
                _ = cancellation.cancelled(), if output.capture.is_none() => {
                    complete_disconnect(
                        &state,
                        &session_id,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    break 'request false;
                }
                result = await_deadline(active.stream.next(), next_deadline) => result,
            };
            let frames = match next {
                Ok(Some(Ok(chunk))) => {
                    active.stats.upstream_bytes = active
                        .stats
                        .upstream_bytes
                        .saturating_add(chunk.len() as u64);
                    match active.decoder.feed(&chunk) {
                        Ok(frames) => frames,
                        Err(error) => {
                            if !active.business_committed {
                                match retry_after_public_start(
                                    &state,
                                    &session_id,
                                    failure_retry_outcome(&active, "protocol_error", &error),
                                )
                                .await
                                {
                                    Ok(RetryResolution::Active(next)) => {
                                        active = next;
                                        continue 'request;
                                    }
                                    Ok(RetryResolution::Final(message)) => {
                                        break 'request send_final_inband(&mut output, &message)
                                            .await
                                            .is_ok();
                                    }
                                    Err(_) => break 'request false,
                                }
                            } else {
                                complete_failure(
                                    &state,
                                    &session_id,
                                    active.plan.attempt_id.clone(),
                                    active.status,
                                    &mut active.stats,
                                    "protocol_error",
                                    &error,
                                )
                                .await;
                            }
                            break 'request false;
                        }
                    }
                }
                Ok(None) => match active.decoder.finish() {
                    Ok(frames) if !frames.is_empty() => frames,
                    Ok(_) => {
                        let detail = "Responses upstream ended without a terminal response event";
                        if !active.business_committed {
                            match retry_after_public_start(
                                &state,
                                &session_id,
                                failure_retry_outcome(&active, "protocol_error", detail),
                            )
                            .await
                            {
                                Ok(RetryResolution::Active(next)) => {
                                    active = next;
                                    continue 'request;
                                }
                                Ok(RetryResolution::Final(message)) => {
                                    break 'request send_final_inband(&mut output, &message)
                                        .await
                                        .is_ok();
                                }
                                Err(_) => break 'request false,
                            }
                        } else {
                            complete_failure(
                                &state,
                                &session_id,
                                active.plan.attempt_id.clone(),
                                active.status,
                                &mut active.stats,
                                "protocol_error",
                                detail,
                            )
                            .await;
                        }
                        break 'request false;
                    }
                    Err(error) => {
                        if !active.business_committed {
                            match retry_after_public_start(
                                &state,
                                &session_id,
                                failure_retry_outcome(&active, "protocol_error", &error),
                            )
                            .await
                            {
                                Ok(RetryResolution::Active(next)) => {
                                    active = next;
                                    continue 'request;
                                }
                                Ok(RetryResolution::Final(message)) => {
                                    break 'request send_final_inband(&mut output, &message)
                                        .await
                                        .is_ok();
                                }
                                Err(_) => break 'request false,
                            }
                        } else {
                            complete_failure(
                                &state,
                                &session_id,
                                active.plan.attempt_id.clone(),
                                active.status,
                                &mut active.stats,
                                "protocol_error",
                                &error,
                            )
                            .await;
                        }
                        break 'request false;
                    }
                },
                Ok(Some(Err(error))) => {
                    let detail = format!("upstream stream read failed: {error}");
                    if !active.business_committed {
                        match retry_after_public_start(
                            &state,
                            &session_id,
                            failure_retry_outcome(&active, "transport_error", &detail),
                        )
                        .await
                        {
                            Ok(RetryResolution::Active(next)) => {
                                active = next;
                                continue 'request;
                            }
                            Ok(RetryResolution::Final(message)) => {
                                break 'request send_final_inband(&mut output, &message)
                                    .await
                                    .is_ok();
                            }
                            Err(_) => break 'request false,
                        }
                    } else {
                        complete_failure(
                            &state,
                            &session_id,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "transport_error",
                            &detail,
                        )
                        .await;
                    }
                    break 'request false;
                }
                Err(error) => {
                    if !active.business_committed {
                        match retry_after_public_start(
                            &state,
                            &session_id,
                            failure_retry_outcome(&active, "transport_error", &error),
                        )
                        .await
                        {
                            Ok(RetryResolution::Active(next)) => {
                                active = next;
                                continue 'request;
                            }
                            Ok(RetryResolution::Final(message)) => {
                                break 'request send_final_inband(&mut output, &message)
                                    .await
                                    .is_ok();
                            }
                            Err(_) => break 'request false,
                        }
                    } else {
                        complete_failure(
                            &state,
                            &session_id,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "transport_error",
                            &error,
                        )
                        .await;
                    }
                    break 'request false;
                }
            };

            match process_active_frames(&state, &session_id, &mut active, &mut output, frames).await
            {
                ActiveFrameResult::Continue => {}
                ActiveFrameResult::Done(cacheable) => break 'request cacheable,
                ActiveFrameResult::Retry(outcome) => {
                    match retry_after_public_start(&state, &session_id, outcome).await {
                        Ok(RetryResolution::Active(next)) => {
                            active = next;
                            continue 'request;
                        }
                        Ok(RetryResolution::Final(message)) => {
                            break 'request send_final_inband(&mut output, &message).await.is_ok();
                        }
                        Err(_) => break 'request false,
                    }
                }
            }
        }
    };
    finished.store(true, Ordering::Release);
    if let Some(task) = control_drain.take() {
        task.abort();
    }
    output.finish(cacheable).await;
}

async fn flush_initial_output(
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
) -> Result<(), ()> {
    let mut wires = std::mem::take(&mut active.early_output);
    if active.business_committed {
        wires.extend(std::mem::take(&mut active.buffered));
    }
    for wire in wires {
        active.stats.observe_wire(&wire);
        output.send_wire(wire).await?;
    }
    Ok(())
}

fn semantic_retry_outcome(active: &ActiveAttempt, terminal: Terminal) -> Value {
    match terminal {
        Terminal::SemanticFailure {
            event_type,
            payload,
        } => {
            let mut outcome = active.stats.report();
            outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
            outcome["kind"] = Value::String("semantic_error".into());
            outcome["status_code"] = Value::from(502);
            outcome["upstream_status_code"] = Value::from(active.status.as_u16());
            outcome["event_type"] = Value::String(event_type);
            outcome["payload"] = payload;
            outcome["committed"] = Value::Bool(false);
            outcome
        }
        _ => failure_retry_outcome(
            active,
            "protocol_error",
            "Responses precommit attempt ended unexpectedly",
        ),
    }
}

fn failure_retry_outcome(active: &ActiveAttempt, kind: &str, detail: &str) -> Value {
    let mut outcome = active.stats.report();
    outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
    outcome["kind"] = Value::String(kind.into());
    outcome["status_code"] = Value::from(if kind == "transport_error" { 504 } else { 502 });
    outcome["upstream_status_code"] = Value::from(active.status.as_u16());
    outcome["detail"] = Value::String(detail.chars().take(4096).collect());
    outcome["committed"] = Value::Bool(false);
    outcome
}

fn commit_observation(active: &ActiveAttempt, reason: &str) -> Value {
    json!({
        "attempt_id": active.plan.attempt_id,
        "upstream_status_code": active.status.as_u16(),
        "commit_reason": reason,
        "business_committed": active.business_committed,
        "precommit_events": active.stats.event_count,
        "precommit_bytes": active.buffered.iter().map(|item| item.len() as u64).sum::<u64>(),
    })
}

enum RetryResolution {
    Active(ActiveAttempt),
    Final(Value),
}

async fn retry_after_public_start(
    state: &AppState,
    session_id: &str,
    mut outcome: Value,
) -> Result<RetryResolution, String> {
    loop {
        let message = control_advance(state, session_id, &outcome).await?;
        if message.get("kind").and_then(Value::as_str) == Some("final") {
            return Ok(RetryResolution::Final(message));
        }
        let plan: Plan = serde_json::from_value(message.clone())
            .map_err(|error| format!("retry plan is invalid: {error}"))?;
        match preflight_attempt(state, plan.clone(), true).await {
            Ok(PreflightResult::Started(active)) => {
                let observation = commit_observation(&active, active.commit_reason);
                control_commit(state, session_id, &observation).await?;
                return Ok(RetryResolution::Active(active));
            }
            Ok(PreflightResult::Retry(mut next_outcome)) => {
                next_outcome["attempt_id"] = Value::String(plan.attempt_id);
                outcome = next_outcome;
            }
            Err(error) => {
                outcome = json!({
                    "attempt_id": plan.attempt_id,
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
            }
        }
    }
}

async fn send_final_inband(output: &mut OutputSink, message: &Value) -> Result<(), ()> {
    if let Some(encoded) = message
        .get("stream_failure_terminal_b64")
        .and_then(Value::as_str)
    {
        if let Ok(wire) = BASE64.decode(encoded) {
            return output.send_wire(Bytes::from(wire)).await;
        }
    }
    let status = message
        .get("status_code")
        .and_then(Value::as_u64)
        .unwrap_or(502);
    let detail = message
        .get("body_b64")
        .and_then(Value::as_str)
        .and_then(|value| BASE64.decode(value).ok())
        .map(|value| String::from_utf8_lossy(&value).chars().take(2048).collect())
        .unwrap_or_else(|| "All Responses providers failed".to_owned());
    let error = json!({
        "type": "error",
        "error": {"message": detail, "status_code": status},
    });
    let wire = encode_event("error", &error).map_err(|_| ())?;
    output.send_wire(wire).await?;
    output
        .send_wire(Bytes::from_static(b"data: [DONE]\n\n"))
        .await
}

enum ActiveFrameResult {
    Continue,
    Done(bool),
    Retry(Value),
}

async fn process_active_frames(
    state: &AppState,
    session_id: &str,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    frames: Vec<SseFrame>,
) -> ActiveFrameResult {
    for frame in frames {
        let processed = match active.processor.process(frame, &mut active.stats) {
            Ok(processed) => processed,
            Err(error) => {
                if !active.business_committed {
                    return ActiveFrameResult::Retry(failure_retry_outcome(
                        active,
                        "protocol_error",
                        &error,
                    ));
                }
                complete_failure(
                    state,
                    session_id,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "protocol_error",
                    &error,
                )
                .await;
                return ActiveFrameResult::Done(false);
            }
        };
        if let Some(Terminal::SemanticFailure {
            event_type,
            payload,
        }) = processed.terminal.as_ref()
        {
            if !active.business_committed {
                let mut outcome = active.stats.report();
                outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
                outcome["kind"] = Value::String("semantic_error".into());
                outcome["status_code"] = Value::from(502);
                outcome["upstream_status_code"] = Value::from(active.status.as_u16());
                outcome["event_type"] = Value::String(event_type.clone());
                outcome["payload"] = payload.clone();
                outcome["committed"] = Value::Bool(false);
                return ActiveFrameResult::Retry(outcome);
            }
            let mut outcome = active.stats.report();
            outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
            outcome["kind"] = Value::String("semantic_failure".into());
            outcome["status_code"] = Value::from(502);
            outcome["upstream_status_code"] = Value::from(active.status.as_u16());
            outcome["event_type"] = Value::String(event_type.clone());
            outcome["payload"] = payload.clone();
            outcome["committed"] = Value::Bool(true);
            let mut terminal_sent = false;
            if let Ok(reply) = control_complete(state, session_id, &outcome).await {
                if let Some(encoded) = reply.get("terminal_b64").and_then(Value::as_str) {
                    if let Ok(terminal) = BASE64.decode(encoded) {
                        let wire = Bytes::from(terminal);
                        active.stats.observe_wire(&wire);
                        terminal_sent = output.send_wire(wire).await.is_ok();
                    }
                }
            }
            return ActiveFrameResult::Done(terminal_sent);
        }

        let suppress_repeated_keepalive = !active.business_committed
            && active.precommit_keepalive_sent
            && processed.event_type.as_deref() == Some("keepalive");
        if let Some(wire) = processed.wire.filter(|_| !suppress_repeated_keepalive) {
            if active.business_committed {
                active.stats.observe_wire(&wire);
                if output.send_wire(wire).await.is_err() {
                    complete_disconnect(
                        state,
                        session_id,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    return ActiveFrameResult::Done(false);
                }
            } else {
                active.buffered.push(wire);
            }
        }

        if processed.commits && !active.business_committed {
            active.business_committed = true;
            let observation = commit_observation(active, "real_output");
            if control_commit(state, session_id, &observation)
                .await
                .is_err()
            {
                return ActiveFrameResult::Done(false);
            }
            for wire in std::mem::take(&mut active.buffered) {
                active.stats.observe_wire(&wire);
                if output.send_wire(wire).await.is_err() {
                    complete_disconnect(
                        state,
                        session_id,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    return ActiveFrameResult::Done(false);
                }
            }
        }
        if let Some(terminal) = processed.terminal {
            finish_terminal(
                state,
                session_id,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
                terminal,
            )
            .await;
            return ActiveFrameResult::Done(true);
        }
    }
    ActiveFrameResult::Continue
}

async fn finish_terminal(
    state: &AppState,
    session_id: &str,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
    terminal: Terminal,
) {
    let kind = match terminal {
        Terminal::Completed => "completed",
        Terminal::Incomplete => "incomplete",
        Terminal::SemanticFailure { .. } => "semantic_failure",
    };
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String(kind.into());
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["committed"] = Value::Bool(true);
    let _ = control_complete(state, session_id, &outcome).await;
}

async fn complete_disconnect(
    state: &AppState,
    session_id: &str,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
) {
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String("downstream_disconnected".into());
    outcome["status_code"] = Value::from(499);
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["committed"] = Value::Bool(true);
    let _ = control_complete(state, session_id, &outcome).await;
}

async fn complete_failure(
    state: &AppState,
    session_id: &str,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
    kind: &str,
    detail: &str,
) {
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String(kind.into());
    outcome["status_code"] = Value::from(502);
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["detail"] = Value::String(detail.chars().take(4096).collect());
    outcome["committed"] = Value::Bool(true);
    let _ = control_complete(state, session_id, &outcome).await;
}

fn downstream_write_timeout() -> Duration {
    static TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        let seconds = std::env::var("RUST_DOWNSTREAM_WRITE_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(30.0);
        Duration::from_secs_f64(seconds)
    })
}

struct GuardedBodyStream {
    inner: ReceiverStream<Result<Bytes, io::Error>>,
    cancellation: CancellationToken,
    finished: Arc<AtomicBool>,
}

impl Stream for GuardedBodyStream {
    type Item = Result<Bytes, io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for GuardedBodyStream {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Acquire) {
            self.cancellation.cancel();
        }
    }
}

#[derive(Debug)]
struct ProcessedEvent {
    wire: Option<Bytes>,
    event_type: Option<String>,
    commits: bool,
    canonical_keepalive: bool,
    terminal: Option<Terminal>,
}

struct ResponsesProcessor {
    commit_policy: String,
    normalizer: Option<CustomIdNormalizer>,
}

impl ResponsesProcessor {
    fn new(commit_policy: String, normalize_ids: bool) -> Self {
        let commit_policy = match commit_policy.trim().to_ascii_lowercase().as_str() {
            "completed_usage" => "completed_usage".to_owned(),
            _ => "real_output".to_owned(),
        };
        Self {
            commit_policy,
            normalizer: normalize_ids.then(CustomIdNormalizer::default),
        }
    }

    fn process(
        &mut self,
        frame: SseFrame,
        stats: &mut StreamStats,
    ) -> Result<ProcessedEvent, String> {
        stats.event_count = stats.event_count.saturating_add(1);
        if frame.comment_only {
            if frame.raw.starts_with(": oaix-terminal-flush-v1 ") {
                return Ok(ProcessedEvent {
                    wire: None,
                    event_type: None,
                    commits: false,
                    canonical_keepalive: false,
                    terminal: None,
                });
            }
            return Ok(ProcessedEvent {
                wire: Some(Bytes::from(format!("{}\n\n", frame.raw))),
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        }
        let Some(data) = frame.data else {
            return Ok(ProcessedEvent {
                wire: None,
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        };
        if data.trim() == "[DONE]" {
            return Err(
                "Responses upstream emitted [DONE] without a terminal response event".into(),
            );
        }
        let mut payload: Value = serde_json::from_str(&data)
            .map_err(|error| format!("Responses upstream event JSON is invalid: {error}"))?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "Responses upstream event payload is missing a string type".to_owned())?
            .to_owned();
        if event_type.len() > 256 || event_type.contains(['\r', '\n']) {
            return Err("Responses upstream event type is invalid".into());
        }
        if let Some(declared) = frame.declared_event.as_deref() {
            if declared != event_type {
                return Err(format!(
                    "Responses event field {declared:?} does not match payload type {event_type:?}"
                ));
            }
        }
        validate_terminal(&event_type, &payload)?;

        let normalized = if let Some(normalizer) = self.normalizer.as_mut() {
            normalizer.normalize(&mut payload)?
        } else {
            false
        };
        let wire = if normalized {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            encode_event(&event_type, &payload)?
        } else if frame.declared_event.is_none() {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            Bytes::from(format!("event: {event_type}\n{}\n\n", frame.raw))
        } else {
            Bytes::from(format!("{}\n\n", frame.raw))
        };
        if event_type.ends_with(".delta") {
            stats.delta_events = stats.delta_events.saturating_add(1);
        }
        if let Some(usage) = extract_usage(&payload) {
            stats.usage = Some(usage.clone());
        }

        let terminal = if semantic_failure(&event_type, &payload) {
            Some(Terminal::SemanticFailure {
                event_type: event_type.clone(),
                payload: payload.clone(),
            })
        } else if event_type == "response.completed" {
            Some(Terminal::Completed)
        } else if event_type == "response.incomplete" {
            Some(Terminal::Incomplete)
        } else {
            None
        };
        let canonical_keepalive = event_type == "keepalive" && is_canonical_keepalive(&payload);
        let commits = terminal.is_some()
            || (!matches!(
                event_type.as_str(),
                "response.created" | "response.in_progress" | "response.queued" | "keepalive"
            ) && self.commit_policy != "completed_usage"
                && has_real_output(&event_type, &payload));
        Ok(ProcessedEvent {
            wire: Some(wire),
            event_type: Some(event_type),
            commits,
            canonical_keepalive,
            terminal,
        })
    }
}

fn validate_terminal(event_type: &str, payload: &Value) -> Result<(), String> {
    if !matches!(
        event_type,
        "error" | "response.completed" | "response.failed" | "response.incomplete"
    ) {
        return Ok(());
    }
    let object = payload
        .as_object()
        .ok_or_else(|| format!("Responses upstream {event_type} payload must be a JSON object"))?;
    if event_type.starts_with("response.") && !object.get("response").is_some_and(Value::is_object)
    {
        return Err(format!(
            "Responses upstream {event_type} payload is missing response"
        ));
    }
    if event_type == "error" && object.get("error").is_none_or(Value::is_null) {
        return Err("Responses upstream error payload is missing error".into());
    }
    Ok(())
}

fn semantic_failure(event_type: &str, payload: &Value) -> bool {
    if matches!(event_type, "error" | "response.failed") {
        return true;
    }
    let status = payload.get("status").and_then(Value::as_str);
    let response_status = payload
        .get("response")
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str);
    status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || response_status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || payload.get("error").is_some_and(Value::is_object)
}

fn is_canonical_keepalive(payload: &Value) -> bool {
    let Some(object) = payload.as_object() else {
        return false;
    };
    object.len() == 2
        && object.get("type").and_then(Value::as_str) == Some("keepalive")
        && object.get("sequence_number").and_then(Value::as_i64) == Some(0)
}

fn has_real_output(event_type: &str, payload: &Value) -> bool {
    if event_type.starts_with("response.") && event_type.ends_with(".delta") {
        return payload
            .get("delta")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty());
    }
    if matches!(
        event_type,
        "response.content_part.added" | "response.content_part.done"
    ) {
        return payload.get("part").is_some_and(part_has_text);
    }
    if event_type == "response.output_item.done" {
        return payload.get("item").is_some_and(item_has_output);
    }
    if event_type.starts_with("response.") && event_type.ends_with(".done") {
        return ["text", "refusal", "arguments"].iter().any(|field| {
            payload
                .get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        });
    }
    false
}

fn part_has_text(part: &Value) -> bool {
    ["text", "refusal"].iter().any(|field| {
        part.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    })
}

fn item_has_output(item: &Value) -> bool {
    if item
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|parts| parts.iter().any(part_has_text))
    {
        return true;
    }
    if !matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "tool_call")
    ) {
        return false;
    }
    ["name", "arguments", "call_id"].iter().any(|field| {
        item.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    })
}

fn extract_usage(payload: &Value) -> Option<&Value> {
    payload
        .get("response")
        .and_then(|value| value.get("usage"))
        .filter(|value| value.is_object())
        .or_else(|| payload.get("usage").filter(|value| value.is_object()))
}

fn encode_event(event_type: &str, payload: &Value) -> Result<Bytes, String> {
    let data = serde_json::to_string(payload)
        .map_err(|error| format!("Responses event encoding failed: {error}"))?;
    Ok(Bytes::from(format!(
        "event: {event_type}\ndata: {data}\n\n"
    )))
}

#[derive(Default)]
struct CustomIdNormalizer {
    id_map: HashMap<String, String>,
    seen_ids: HashSet<String>,
}

impl CustomIdNormalizer {
    fn normalize(&mut self, payload: &mut Value) -> Result<bool, String> {
        let Some(root) = payload.as_object_mut() else {
            return Ok(false);
        };
        let existing = collect_item_ids(root);
        let mut changed = false;
        if let Some(item) = root.get_mut("item").and_then(Value::as_object_mut) {
            changed |= self.normalize_item(item, &existing)?;
        }
        for name in ["input", "output"] {
            if let Some(items) = root.get_mut(name).and_then(Value::as_array_mut) {
                for item in items.iter_mut().filter_map(Value::as_object_mut) {
                    changed |= self.normalize_item(item, &existing)?;
                }
            }
        }
        if let Some(items) = root
            .get_mut("response")
            .and_then(Value::as_object_mut)
            .and_then(|response| response.get_mut("output"))
            .and_then(Value::as_array_mut)
        {
            for item in items.iter_mut().filter_map(Value::as_object_mut) {
                changed |= self.normalize_item(item, &existing)?;
            }
        }

        let event_type = root.get("type").and_then(Value::as_str).unwrap_or_default();
        let item_reference = root
            .get("item_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if matches!(
            event_type,
            "response.custom_tool_call_input.delta" | "response.custom_tool_call_input.done"
        ) {
            if let Some(reference) = item_reference.as_deref() {
                if let Some(normalized) = normalized_id(reference) {
                    self.ensure_no_collision(reference, &normalized, &existing)?;
                    self.id_map.insert(reference.to_owned(), normalized);
                }
            }
        }
        if let Some(reference) = item_reference {
            if let Some(normalized) = self.id_map.get(&reference).cloned() {
                root.insert("item_id".into(), Value::String(normalized));
                changed = true;
            }
        }
        self.seen_ids.extend(collect_item_ids(root));
        Ok(changed)
    }

    fn normalize_item(
        &mut self,
        item: &mut Map<String, Value>,
        existing: &HashSet<String>,
    ) -> Result<bool, String> {
        if item.get("type").and_then(Value::as_str) != Some("custom_tool_call") {
            return Ok(false);
        }
        let Some(item_id) = item.get("id").and_then(Value::as_str).map(str::to_owned) else {
            return Ok(false);
        };
        let Some(normalized) = normalized_id(&item_id) else {
            return Ok(false);
        };
        self.ensure_no_collision(&item_id, &normalized, existing)?;
        self.id_map.insert(item_id, normalized.clone());
        item.insert("id".into(), Value::String(normalized));
        Ok(true)
    }

    fn ensure_no_collision(
        &self,
        original: &str,
        normalized: &str,
        existing: &HashSet<String>,
    ) -> Result<(), String> {
        if normalized != original
            && (existing.contains(normalized)
                || (self.seen_ids.contains(normalized)
                    && self.id_map.get(original).map(String::as_str) != Some(normalized)))
        {
            return Err(
                "custom_tool_call ID normalization would collide with an existing item ID".into(),
            );
        }
        Ok(())
    }
}

fn normalized_id(value: &str) -> Option<String> {
    let suffix = value.strip_prefix("item_")?;
    (!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric()))
        .then(|| format!("ctc_{suffix}"))
}

fn collect_item_ids(root: &Map<String, Value>) -> HashSet<String> {
    let mut ids = HashSet::new();
    if let Some(item) = root.get("item").and_then(Value::as_object) {
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            ids.insert(id.to_owned());
        }
    }
    for name in ["input", "output"] {
        if let Some(items) = root.get(name).and_then(Value::as_array) {
            for item in items.iter().filter_map(Value::as_object) {
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    ids.insert(id.to_owned());
                }
            }
        }
    }
    if let Some(items) = root
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("output"))
        .and_then(Value::as_array)
    {
        for item in items.iter().filter_map(Value::as_object) {
            if let Some(id) = item.get("id").and_then(Value::as_str) {
                ids.insert(id.to_owned());
            }
        }
    }
    ids
}

#[derive(Debug)]
struct SseFrame {
    raw: String,
    declared_event: Option<String>,
    data: Option<String>,
    comment_only: bool,
}

struct SseDecoder {
    buffer: Vec<u8>,
    max_event_bytes: usize,
}

impl SseDecoder {
    fn new(max_event_bytes: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_event_bytes: max_event_bytes.max(1),
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some((end, delimiter_len)) = find_event_delimiter(&self.buffer) {
            if end > self.max_event_bytes {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let raw = self.buffer[..end].to_vec();
            self.buffer.drain(..end + delimiter_len);
            if raw.iter().all(|byte| byte.is_ascii_whitespace()) {
                continue;
            }
            frames.push(parse_sse_frame(&raw)?);
        }
        if self.buffer.len() > self.max_event_bytes.saturating_add(64 * 1024) {
            return Err("Responses upstream SSE pending frame exceeds the configured limit".into());
        }
        Ok(frames)
    }

    fn finish(&mut self) -> Result<Vec<SseFrame>, String> {
        let mut frames = self.feed(&[])?;
        if !self.buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            if self.buffer.len() > self.max_event_bytes {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let raw = std::mem::take(&mut self.buffer);
            frames.push(parse_sse_frame(&raw)?);
        }
        self.buffer.clear();
        Ok(frames)
    }
}

fn find_event_delimiter(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") || bytes[index..].starts_with(b"\r\r") {
            return Some((index, 2));
        }
        index += 1;
    }
    None
}

fn parse_sse_frame(raw: &[u8]) -> Result<SseFrame, String> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| "Responses upstream SSE event is not valid UTF-8".to_owned())?;
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut declared_event = None;
    let mut data_lines = Vec::new();
    let mut saw_field = false;
    let mut saw_comment = false;
    for line in normalized.split('\n') {
        if let Some(_comment) = line.strip_prefix(':') {
            saw_comment = true;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        saw_field = true;
        let (field, mut value) = line.split_once(':').unwrap_or((line, ""));
        if let Some(stripped) = value.strip_prefix(' ') {
            value = stripped;
        }
        match field {
            "event" => declared_event = Some(value.to_owned()),
            "data" => data_lines.push(value.to_owned()),
            _ => {}
        }
    }
    Ok(SseFrame {
        raw: normalized,
        declared_event,
        data: (!data_lines.is_empty()).then(|| data_lines.join("\n")),
        comment_only: saw_comment && !saw_field,
    })
}

async fn control_get_plan(state: &AppState, session_id: &str) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::GET,
        &format!("/_internal/rust-responses/{session_id}/plan"),
        None,
    )
    .await
}

async fn control_advance(
    state: &AppState,
    session_id: &str,
    outcome: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/advance"),
        Some(outcome),
    )
    .await
}

async fn control_commit(
    state: &AppState,
    session_id: &str,
    observation: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/commit"),
        Some(observation),
    )
    .await
}

async fn control_complete(
    state: &AppState,
    session_id: &str,
    outcome: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/complete"),
        Some(outcome),
    )
    .await
}

async fn control_request(
    state: &AppState,
    method: reqwest::Method,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let mut request = state
        .backend_client
        .request(method, state.internal_url(path))
        .header(CONTROL_HEADER, state.control_token.as_ref());
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("control request failed: {error}"))?;
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("control response read failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "control response status {}: {}",
            status.as_u16(),
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(1024)
                .collect::<String>()
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("control response JSON is invalid: {error}"))
}

async fn response_from_final(
    message: &Value,
    control_headers: &HeaderMap,
    owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let status = message
        .get("status_code")
        .and_then(Value::as_u64)
        .and_then(|value| StatusCode::from_u16(value as u16).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let body = Bytes::from(
        message
            .get("body_b64")
            .and_then(Value::as_str)
            .and_then(|value| BASE64.decode(value).ok())
            .unwrap_or_default(),
    );
    let mut headers = HeaderMap::new();
    if let Some(values) = message.get("headers").and_then(Value::as_object) {
        for (name, value) in values {
            let (Ok(name), Some(value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                value
                    .as_str()
                    .and_then(|value| HeaderValue::from_str(value).ok()),
            ) else {
                continue;
            };
            if name.as_str().eq_ignore_ascii_case("content-length") {
                continue;
            }
            headers.append(name, value);
        }
    }
    for (name, value) in control_headers {
        headers.insert(name.clone(), value.clone());
    }
    if let Some(owner) = owner {
        if body.len() > owner.max_response_bytes() {
            owner.nonreplayable("response_too_large").await;
        } else {
            owner
                .complete(status, headers.clone(), vec![body.clone()], body.len())
                .await;
        }
        return idempotency::response_from_bytes(status, headers, body);
    }
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

async fn release_owner(owner: &mut Option<idempotency::Owner>) {
    if let Some(owner) = owner.take() {
        owner.release().await;
    }
}

fn public_control_headers(headers: &HeaderMap) -> HeaderMap {
    let mut selected = HeaderMap::new();
    for (name, value) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if lower == "x-request-id" || lower.starts_with("access-control-") {
            selected.append(name.clone(), value.clone());
        }
    }
    selected
}

async fn read_limited_body(
    response: reqwest::Response,
    deadline: Option<tokio::time::Instant>,
) -> String {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while body.len() < MAX_ERROR_BODY_BYTES {
        match await_deadline(stream.next(), deadline).await {
            Ok(Some(Ok(chunk))) => {
                let remaining = MAX_ERROR_BODY_BYTES - body.len();
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            _ => break,
        }
    }
    String::from_utf8_lossy(&body).into_owned()
}

fn deadline(started: tokio::time::Instant, seconds: Option<f64>) -> Option<tokio::time::Instant> {
    seconds
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| started + Duration::from_secs_f64(value))
}

fn earlier_deadline(
    first: Option<tokio::time::Instant>,
    second: Option<tokio::time::Instant>,
) -> Option<tokio::time::Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

async fn await_deadline<F, T>(
    future: F,
    deadline: Option<tokio::time::Instant>,
) -> Result<T, String>
where
    F: std::future::Future<Output = T>,
{
    if let Some(deadline) = deadline {
        tokio::time::timeout_at(deadline, future)
            .await
            .map_err(|_| "upstream deadline exceeded".to_owned())
    } else {
        Ok(future.await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_active(engine: &str) -> ActiveAttempt {
        let plan = Plan {
            attempt_id: "attempt-1".into(),
            url: "http://provider.example/v1/responses".into(),
            headers: HashMap::new(),
            body: "{}".into(),
            proxy: None,
            engine: engine.into(),
            http1_only: false,
            commit_policy: "real_output".into(),
            normalize_custom_tool_call_ids: false,
            first_byte_timeout_seconds: None,
            idle_timeout_seconds: None,
            total_timeout_seconds: None,
            max_event_bytes: 4096,
            max_precommit_items: 128,
            max_precommit_bytes: 64 * 1024,
        };
        ActiveAttempt {
            decoder: SseDecoder::new(plan.max_event_bytes),
            processor: ResponsesProcessor::new("real_output".into(), false),
            plan,
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            stream: Box::pin(futures_util::stream::empty()),
            early_output: Vec::new(),
            buffered: Vec::new(),
            stats: StreamStats::default(),
            terminal: None,
            total_deadline: None,
            commit_reason: "real_output",
            precommit_keepalive_sent: false,
            business_committed: false,
        }
    }

    #[test]
    fn decoder_preserves_fragmented_canonical_frames() {
        let mut decoder = SseDecoder::new(1024);
        assert!(decoder.feed(b"event: response.output_").unwrap().is_empty());
        let frames = decoder
            .feed(
                b"text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n",
            )
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].declared_event.as_deref(),
            Some("response.output_text.delta")
        );
    }

    #[test]
    fn processor_canonicalizes_data_only_events() {
        let mut decoder = SseDecoder::new(1024);
        let frame = decoder
            .feed(b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let mut processor = ResponsesProcessor::new("real_output".into(), false);
        let mut stats = StreamStats::default();
        let event = processor.process(frame, &mut stats).unwrap();
        assert!(event.commits);
        assert!(event
            .wire
            .unwrap()
            .starts_with(b"event: response.output_text.delta\n"));
    }

    #[test]
    fn custom_tool_call_ids_are_statefully_rewritten() {
        let mut processor = ResponsesProcessor::new("real_output".into(), true);
        let mut stats = StreamStats::default();
        let mut decoder = SseDecoder::new(4096);
        let added = decoder
            .feed(b"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"custom_tool_call\",\"id\":\"item_abc\"}}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let first = processor.process(added, &mut stats).unwrap().wire.unwrap();
        assert!(String::from_utf8_lossy(&first).contains("ctc_abc"));
        let delta = decoder
            .feed(b"event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"item_abc\",\"delta\":\"{}\"}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let second = processor.process(delta, &mut stats).unwrap().wire.unwrap();
        assert!(String::from_utf8_lossy(&second).contains("ctc_abc"));
    }

    #[test]
    fn response_failed_is_a_semantic_terminal() {
        let mut decoder = SseDecoder::new(4096);
        let frame = decoder
            .feed(b"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"nope\"}}}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let mut processor = ResponsesProcessor::new("real_output".into(), false);
        let mut stats = StreamStats::default();
        assert!(matches!(
            processor.process(frame, &mut stats).unwrap().terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }

    #[test]
    fn preflight_keeps_every_frame_after_commit_in_the_same_chunk() {
        let mut active = test_active("gpt");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"b\"}\n\n\
                  event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected a committed stream");
        };
        assert!(active.business_committed);
        assert!(matches!(active.terminal, Some(Terminal::Completed)));
        assert_eq!(
            active.buffered.len(),
            5,
            "synthetic keepalive + four frames"
        );
    }

    #[test]
    fn codex_keepalive_starts_http_but_keeps_semantic_failure_retryable() {
        let mut active = test_active("codex");
        let frames = active
            .decoder
            .feed(
                b"event: keepalive\ndata: {\"type\":\"keepalive\",\"sequence_number\":0}\n\n\
                  event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"retry me\"}}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected keepalive-started stream");
        };
        assert!(!active.business_committed);
        assert_eq!(active.early_output.len(), 1);
        assert_eq!(active.buffered.len(), 1, "response.created remains private");
        assert!(matches!(
            active.terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }
}
