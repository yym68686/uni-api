use crate::protocols::responses::events::has_real_output;
use crate::protocols::responses::events::item_has_output;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

pub(crate) struct StreamStats {
    pub(crate) transport_failed: bool,
    pub(crate) stream_mode: &'static str,
    pub(crate) upstream_bytes: u64,
    pub(crate) upstream_chunks: u64,
    pub(crate) downstream_bytes: u64,
    pub(crate) downstream_chunks: u64,
    pub(crate) event_count: u64,
    pub(crate) delta_events: u64,
    pub(crate) normalized_events: u64,
    pub(crate) usage: Option<Value>,
    pub(crate) wire_hash: Option<Sha256>,
    pub(crate) started_at: tokio::time::Instant,
    pub(crate) first_output_ms: Option<f64>,
    pub(crate) response_created_ms: Option<f64>,
    pub(crate) first_text_ms: Option<f64>,
    pub(crate) headers_received_ms: Option<f64>,
    pub(crate) first_upstream_chunk_ms: Option<f64>,
    pub(crate) public_stream_ready_ms: Option<f64>,
    pub(crate) first_wire_prepared_ms: Option<f64>,
    pub(crate) preflight_decode_ms: f64,
    pub(crate) preflight_read_wait_ms: f64,
    pub(crate) preflight_read_calls: u64,
    pub(crate) preflight_process_ms: f64,
    pub(crate) error_body_read_ms: Option<f64>,
    pub(crate) upstream_http_version: Option<String>,
    pub(crate) connection: Option<UpstreamConnection>,
    pub(crate) raw_stream: Option<Box<RawStreamTiming>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransportTiming {
    pub schema: u8,
    pub origin: String,
    pub headers_received_ms: Option<f64>,
    pub first_upstream_chunk_ms: Option<f64>,
    pub public_stream_ready_ms: Option<f64>,
    pub first_wire_prepared_ms: Option<f64>,
    pub preflight_decode_ms: f64,
    #[serde(default)]
    pub preflight_read_wait_ms: Option<f64>,
    #[serde(default)]
    pub preflight_read_calls: Option<u64>,
    #[serde(default)]
    pub preflight_process_ms: Option<f64>,
    #[serde(default)]
    pub error_body_read_ms: Option<f64>,
    #[serde(default)]
    pub raw_stream: Option<Box<RawStreamTiming>>,
    #[serde(default)]
    pub upstream_http_version: Option<String>,
    #[serde(default)]
    pub connection: Option<UpstreamConnection>,
    pub network_write_measured: bool,
}

/// Physical outbound connection only; never the caller's address or HTTP headers.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpstreamConnection {
    pub local_addr: String,
    pub remote_addr: String,
}

/// Fixed-size metadata only; checkpoints are taken after parsing one received chunk.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RawStageTotals {
    pub upstream_read_wait_ms: f64,
    pub upstream_read_calls: u64,
    pub process_ms: f64,
    pub output_send_ms: f64,
    pub output_calls: u64,
    pub frames: u64,
    pub comment_frames: u64,
    pub max_pending_frame_bytes: usize,
    pub last_chunk_ms: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawStreamTiming {
    pub entered_ms: f64,
    pub totals: RawStageTotals,
    pub at_response_created: Option<RawStageTotals>,
    pub at_first_output: Option<RawStageTotals>,
    pub at_first_text: Option<RawStageTotals>,
}

impl StreamStats {
    pub(crate) fn new(attempt_id: &str) -> Self {
        let sample_bps = wire_hash_sample_bps();
        let mut sampler = DefaultHasher::new();
        attempt_id.hash(&mut sampler);
        let sampled = sample_bps >= 10_000 || sampler.finish() % 10_000 < sample_bps;
        Self {
            transport_failed: false,
            stream_mode: "unknown",
            upstream_bytes: 0,
            upstream_chunks: 0,
            downstream_bytes: 0,
            downstream_chunks: 0,
            event_count: 0,
            delta_events: 0,
            normalized_events: 0,
            usage: None,
            wire_hash: sampled.then(Sha256::new),
            started_at: tokio::time::Instant::now(),
            first_output_ms: None,
            response_created_ms: None,
            first_text_ms: None,
            headers_received_ms: None,
            first_upstream_chunk_ms: None,
            public_stream_ready_ms: None,
            first_wire_prepared_ms: None,
            preflight_decode_ms: 0.0,
            preflight_read_wait_ms: 0.0,
            preflight_read_calls: 0,
            preflight_process_ms: 0.0,
            error_body_read_ms: None,
            upstream_http_version: None,
            connection: None,
            raw_stream: None,
        }
    }

    pub(crate) fn report(&self) -> Value {
        let hash = self
            .wire_hash
            .as_ref()
            .map(|hasher| format!("{:x}", hasher.clone().finalize()));
        json!({
            "upstream_bytes": self.upstream_bytes,
            "upstream_chunks": self.upstream_chunks,
            "downstream_bytes": self.downstream_bytes,
            "downstream_chunks": self.downstream_chunks,
            "event_count": self.event_count,
            "delta_events": self.delta_events,
            "normalized_events": self.normalized_events,
            "usage": self.usage,
            "wire_sha256": hash,
            "wire_hash_sampled": self.wire_hash.is_some(),
            "stream_mode": self.stream_mode,
            "first_output_ms": self.first_output_ms,
            "response_created_ms": self.response_created_ms,
            "first_text_ms": self.first_text_ms,
            "transport_timing": self.transport_timing(),
        })
    }

    pub(crate) fn observe_semantic_output(&mut self, event_type: &str, payload: &Value) {
        let elapsed = self.started_at.elapsed().as_secs_f64() * 1000.0;
        if event_type == "response.created" && self.response_created_ms.is_none() {
            self.response_created_ms = Some(elapsed);
        }
        if event_type == "response.output_text.delta"
            && self.first_text_ms.is_none()
            && payload
                .get("delta")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        {
            self.first_text_ms = Some(elapsed);
        }
        if self.first_output_ms.is_none()
            && (has_real_output(event_type, payload)
                || matches!(event_type, "response.completed" | "response.incomplete")
                    && payload
                        .pointer("/response/output")
                        .and_then(Value::as_array)
                        .is_some_and(|items| items.iter().any(item_has_output)))
        {
            self.first_output_ms = Some(self.started_at.elapsed().as_secs_f64() * 1000.0);
        }
    }

    pub(crate) fn observe_upstream(&mut self, chunk: &[u8]) {
        if !chunk.is_empty() {
            if let Some(raw) = self.raw_stream.as_mut() {
                raw.totals.last_chunk_ms = Some(self.started_at.elapsed().as_secs_f64() * 1000.0);
            }
        }
        if !chunk.is_empty() && self.first_upstream_chunk_ms.is_none() {
            self.first_upstream_chunk_ms = Some(self.started_at.elapsed().as_secs_f64() * 1000.0);
        }
        self.upstream_bytes = self.upstream_bytes.saturating_add(chunk.len() as u64);
        self.upstream_chunks = self.upstream_chunks.saturating_add(1);
    }

    pub(crate) fn observe_wire(&mut self, wire: &[u8]) {
        if !wire.is_empty() && self.first_wire_prepared_ms.is_none() {
            self.first_wire_prepared_ms = Some(self.started_at.elapsed().as_secs_f64() * 1000.0);
        }
        self.downstream_bytes = self.downstream_bytes.saturating_add(wire.len() as u64);
        self.downstream_chunks = self.downstream_chunks.saturating_add(1);
        if let Some(hasher) = self.wire_hash.as_mut() {
            hasher.update(wire);
        }
    }

    pub(crate) fn observe_response_connection(&mut self, response: &reqwest::Response) {
        self.upstream_http_version = Some(format!("{:?}", response.version()));
        self.connection = response
            .extensions()
            .get::<hyper_util::client::legacy::connect::HttpInfo>()
            .map(|info| UpstreamConnection {
                local_addr: info.local_addr().to_string(),
                remote_addr: info.remote_addr().to_string(),
            });
    }

    pub(crate) fn begin_raw_observation(&mut self) {
        if self.raw_stream.is_none() {
            self.raw_stream = Some(Box::new(RawStreamTiming {
                entered_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
                totals: RawStageTotals::default(),
                at_response_created: None,
                at_first_output: None,
                at_first_text: None,
            }));
        }
    }

    pub(crate) fn raw_read_wait(&mut self, elapsed_ms: f64) {
        if let Some(raw) = self.raw_stream.as_mut() {
            raw.totals.upstream_read_wait_ms += elapsed_ms;
            raw.totals.upstream_read_calls = raw.totals.upstream_read_calls.saturating_add(1);
        }
    }

    pub(crate) fn raw_processed(&mut self, elapsed_ms: f64, pending_bytes: usize) {
        if let Some(raw) = self.raw_stream.as_mut() {
            raw.totals.process_ms += elapsed_ms;
            raw.totals.max_pending_frame_bytes =
                raw.totals.max_pending_frame_bytes.max(pending_bytes);
            // A milestone already seen during preflight is outside this observation.
            if self
                .response_created_ms
                .is_some_and(|ms| ms >= raw.entered_ms)
                && raw.at_response_created.is_none()
            {
                raw.at_response_created = Some(raw.totals.clone());
            }
            if self.first_output_ms.is_some_and(|ms| ms >= raw.entered_ms)
                && raw.at_first_output.is_none()
            {
                raw.at_first_output = Some(raw.totals.clone());
            }
            if self.first_text_ms.is_some_and(|ms| ms >= raw.entered_ms)
                && raw.at_first_text.is_none()
            {
                raw.at_first_text = Some(raw.totals.clone());
            }
        }
    }

    pub(crate) fn raw_output_wait(&mut self, elapsed_ms: f64) {
        if let Some(raw) = self.raw_stream.as_mut() {
            raw.totals.output_send_ms += elapsed_ms;
            raw.totals.output_calls = raw.totals.output_calls.saturating_add(1);
        }
    }

    pub(crate) fn transport_timing(&self) -> TransportTiming {
        TransportTiming {
            schema: 1,
            origin: "attempt_http_send".into(),
            headers_received_ms: self.headers_received_ms,
            first_upstream_chunk_ms: self.first_upstream_chunk_ms,
            public_stream_ready_ms: self.public_stream_ready_ms,
            first_wire_prepared_ms: self.first_wire_prepared_ms,
            preflight_decode_ms: self.preflight_decode_ms,
            preflight_read_wait_ms: Some(self.preflight_read_wait_ms),
            preflight_read_calls: Some(self.preflight_read_calls),
            preflight_process_ms: Some(self.preflight_process_ms),
            error_body_read_ms: self.error_body_read_ms,
            upstream_http_version: self.upstream_http_version.clone(),
            connection: self.connection.clone(),
            raw_stream: self.raw_stream.clone(),
            network_write_measured: false,
        }
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self::new("")
    }
}

pub(crate) fn wire_hash_sample_bps() -> u64 {
    static SAMPLE_BPS: OnceLock<u64> = OnceLock::new();
    *SAMPLE_BPS.get_or_init(|| {
        std::env::var("RUST_RESPONSES_WIRE_HASH_SAMPLE_BPS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(100)
            .min(10_000)
    })
}

#[cfg(test)]
mod stage_tests {
    use super::*;

    #[test]
    fn raw_milestones_are_one_shot_metadata_and_do_not_reuse_preflight_events() {
        let mut stats = StreamStats::new("metadata");
        assert!(stats.transport_timing().raw_stream.is_none());
        stats.response_created_ms = Some(0.0);
        stats.begin_raw_observation();
        stats.raw_read_wait(300.0);
        stats.raw_output_wait(25.0);
        stats.observe_semantic_output(
            "response.output_text.delta",
            &json!({"delta":"PRIVATE_BODY"}),
        );
        stats.raw_processed(0.2, 100);
        stats.raw_read_wait(500.0);
        stats.raw_processed(0.1, 10);
        let raw = stats.raw_stream.as_ref().unwrap();
        assert!(raw.at_response_created.is_none());
        assert_eq!(
            raw.at_first_text.as_ref().unwrap().upstream_read_wait_ms,
            300.0
        );
        assert_eq!(raw.at_first_text.as_ref().unwrap().output_send_ms, 25.0);
        assert_eq!(raw.totals.upstream_read_wait_ms, 800.0);
        assert_eq!(raw.totals.max_pending_frame_bytes, 100);
        assert!(!serde_json::to_string(&stats.transport_timing())
            .unwrap()
            .contains("PRIVATE_BODY"));
    }

    #[test]
    fn missing_stages_are_null_and_empty_chunks_do_not_create_samples() {
        let mut stats = StreamStats::new("fixture");
        stats.observe_upstream(b"");
        stats.observe_wire(b"");
        let timing = serde_json::to_value(stats.transport_timing()).unwrap();
        for name in [
            "headers_received_ms",
            "first_upstream_chunk_ms",
            "public_stream_ready_ms",
            "first_wire_prepared_ms",
        ] {
            assert!(timing[name].is_null());
        }
        assert_eq!(timing["network_write_measured"], false);
        assert!(stats.first_output_ms.is_none());
    }

    #[test]
    fn transport_samples_are_monotonic_one_shot_and_not_semantic_output() {
        let mut stats = StreamStats::new("fixture");
        stats.started_at = tokio::time::Instant::now() - std::time::Duration::from_millis(50);
        stats.observe_upstream(b"fragment");
        stats.observe_wire(b"fragment");
        let first = serde_json::to_value(stats.transport_timing()).unwrap();
        stats.started_at = tokio::time::Instant::now() - std::time::Duration::from_secs(1);
        stats.observe_upstream(b"later");
        stats.observe_wire(b"later");
        let later = serde_json::to_value(stats.transport_timing()).unwrap();
        assert_eq!(
            first["first_upstream_chunk_ms"],
            later["first_upstream_chunk_ms"]
        );
        assert_eq!(
            first["first_wire_prepared_ms"],
            later["first_wire_prepared_ms"]
        );
        assert!(later["first_upstream_chunk_ms"].as_f64().unwrap() >= 50.0);
        assert!(stats.first_output_ms.is_none());
        assert!(stats.first_text_ms.is_none());
    }
}
