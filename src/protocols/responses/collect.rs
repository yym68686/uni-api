//! Bounded collection of a Responses event stream into its authoritative terminal object.
//! No detached task: cancellation drops the upstream request and its buffers.
use crate::observability::stream::StreamStats;
use crate::protocols::responses::{
    events::{responses_semantic_error, Terminal},
    processor::ResponsesProcessor,
};
use crate::protocols::sse::{parse_sse_frame, SseDecoder, SseFrame};
use crate::transport::body::{upstream_chunks, upstream_response_max_bytes};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Duration;

pub(crate) struct Collected {
    pub payload: Value,
    pub outcome: Value,
}

pub(crate) struct CollectionFailure {
    pub status: u16,
    pub detail: String,
    pub outcome: Value,
}

struct Collector {
    processor: ResponsesProcessor,
    stats: StreamStats,
}

impl Collector {
    fn error(&self, status: u16, kind: &str, detail: String) -> CollectionFailure {
        CollectionFailure {
            status,
            detail: detail.clone(),
            outcome: json!({
                "kind":kind, "status_code":status, "upstream_status_code":200, "detail":detail,
                "retry_allowed":self.stats.first_output_ms.is_none(), "usage":self.stats.usage,
                "upstream_stream":true, "downstream_stream":false,
                "first_output_ms":self.stats.first_output_ms, "first_text_ms":self.stats.first_text_ms,
                "response_created_ms":self.stats.response_created_ms,
            }),
        }
    }

    fn frame(&mut self, frame: SseFrame) -> Result<Option<Collected>, CollectionFailure> {
        // Parse before moving the frame; normalization of the final object follows below.
        let data = parse_sse_frame(&frame)
            .map_err(|e| self.error(502, "protocol_error", e))?
            .data;
        let event = self
            .processor
            .process(frame, &mut self.stats)
            .map_err(|e| self.error(502, "protocol_error", e))?;
        let Some(terminal) = event.terminal else {
            return Ok(None);
        };
        if let Terminal::SemanticFailure {
            event_type,
            payload,
        } = terminal
        {
            let (status, detail) = responses_semantic_error(&payload, &event_type);
            let mut failure = self.error(status, "semantic_failure", detail);
            // Preserve the same structured pre-output evidence as native SSE.
            // The routing classifier must not guess error fields from detail.
            failure.outcome["payload"] = payload;
            failure.outcome["event_type"] = event_type.into();
            failure.outcome["committed"] = self.stats.first_output_ms.is_some().into();
            return Err(failure);
        }
        let mut event: Value = serde_json::from_str(data.as_deref().unwrap_or_default())
            .map_err(|e| self.error(502, "protocol_error", e.to_string()))?;
        let mut payload = event["response"].take();
        let kind = if matches!(terminal, Terminal::Incomplete) {
            "incomplete"
        } else {
            "completed"
        };
        if payload.get("status").and_then(Value::as_str) != Some(kind)
            || !payload.get("output").is_some_and(Value::is_array)
            || payload
                .get("id")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || payload.get("error").is_some_and(|v| !v.is_null())
        {
            return Err(self.error(
                502,
                "protocol_error",
                "Responses terminal is missing a complete, consistent response object".into(),
            ));
        }
        super::usage::normalize_payload(&mut payload);
        if let Some(normalizer) = self.processor.normalizer.as_mut() {
            normalizer
                .normalize(&mut payload)
                .map_err(|e| self.error(502, "protocol_error", e))?;
        }
        Ok(Some(Collected {
            outcome: json!({
                "kind":kind, "status_code":200, "upstream_status_code":200,
                "upstream_stream":true, "downstream_stream":false, "usage":payload.get("usage"),
                "first_output_ms":self.stats.first_output_ms, "first_text_ms":self.stats.first_text_ms,
                "response_created_ms":self.stats.response_created_ms,
            }),
            payload,
        }))
    }
}

pub(crate) async fn collect(
    response: reqwest::Response,
    started: tokio::time::Instant,
    idle: Option<Duration>,
    normalize_ids: bool,
) -> Result<Collected, CollectionFailure> {
    let mut collector = Collector {
        processor: ResponsesProcessor::new("real_output".into(), normalize_ids),
        stats: StreamStats::new("nonstream-collection"),
    };
    collector.stats.started_at = started;
    if !response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        })
    {
        return Err(collector.error(
            502,
            "protocol_error",
            "Responses stream-only upstream did not return text/event-stream".into(),
        ));
    }
    let maximum = upstream_response_max_bytes();
    let mut decoder = SseDecoder::new(maximum);
    let mut received = 0usize;
    let stream = upstream_chunks(response, idle);
    futures_util::pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| collector.error(502, "transport_error", e.to_string()))?;
        received = received.saturating_add(chunk.len());
        if received > maximum {
            return Err(collector.error(
                502,
                "protocol_error",
                format!("Responses collection exceeds the configured {maximum} byte limit"),
            ));
        }
        for frame in decoder
            .feed(&chunk)
            .map_err(|e| collector.error(502, "protocol_error", e))?
        {
            if let Some(done) = collector.frame(frame)? {
                return Ok(done);
            }
        }
    }
    for frame in decoder
        .finish()
        .map_err(|e| collector.error(502, "protocol_error", e))?
    {
        if let Some(done) = collector.frame(frame)? {
            return Ok(done);
        }
    }
    Err(collector.error(
        502,
        "protocol_error",
        "Responses upstream ended without a terminal response event".into(),
    ))
}
