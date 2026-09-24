use crate::protocols::responses::events::has_real_output;
use crate::protocols::responses::events::item_has_output;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

pub(crate) struct StreamStats {
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
}

impl StreamStats {
    pub(crate) fn new(attempt_id: &str) -> Self {
        let sample_bps = wire_hash_sample_bps();
        let mut sampler = DefaultHasher::new();
        attempt_id.hash(&mut sampler);
        let sampled = sample_bps >= 10_000 || sampler.finish() % 10_000 < sample_bps;
        Self {
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
        self.upstream_bytes = self.upstream_bytes.saturating_add(chunk.len() as u64);
        self.upstream_chunks = self.upstream_chunks.saturating_add(1);
    }

    pub(crate) fn observe_wire(&mut self, wire: &[u8]) {
        self.downstream_bytes = self.downstream_bytes.saturating_add(wire.len() as u64);
        self.downstream_chunks = self.downstream_chunks.saturating_add(1);
        if let Some(hasher) = self.wire_hash.as_mut() {
            hasher.update(wire);
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
