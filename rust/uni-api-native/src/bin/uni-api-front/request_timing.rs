//! Process-local arrival and dispatch observations; never trust caller timestamps.
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use serde_json::json;

use crate::channel_metrics::{ChannelMetrics, MetricKey};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestArrival(Instant);

impl RequestArrival {
    pub(crate) fn now() -> Self {
        Self(Instant::now())
    }

    pub(crate) fn attempt(
        self,
        key: MetricKey,
        request_id: String,
        attempt_id: String,
    ) -> AttemptDispatch {
        AttemptDispatch {
            arrival: self,
            key,
            request_id,
            attempt_id,
            recorded: Arc::new(OnceLock::new()),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AttemptDispatch {
    arrival: RequestArrival,
    key: MetricKey,
    request_id: String,
    attempt_id: String,
    recorded: Arc<OnceLock<f64>>,
}

impl AttemptDispatch {
    /// Call immediately before the model HTTP send. Shared clones record once,
    /// even for hedged plans; merely preparing/skipping a plan records nothing.
    pub(crate) fn record(&self, metrics: &ChannelMetrics) -> f64 {
        *self.recorded.get_or_init(|| {
            let elapsed = self.arrival.0.elapsed().as_secs_f64() * 1000.0;
            metrics.observe_dispatch(&self.key, elapsed);
            if let Some(writer) = crate::facts_s3::global() {
                writer.enqueue(crate::facts_s3::dispatch_event(
                    &self.key,
                    &self.request_id,
                    &self.attempt_id,
                    elapsed,
                ));
            }
            eprintln!(
                "{}",
                json!({
                    "kind":"log", "fugue_table":"app_events", "event":"channel_dispatch",
                    "event_type":"channel_dispatch", "severity":"info", "source":"uni-api-ember",
                    "message":"uni-api channel request dispatch", "request_id":self.request_id,
                    "attempt_id":self.attempt_id, "provider":self.key.provider,
                    "model":self.key.model, "actual_model":self.key.upstream_model,
                    "path":self.key.endpoint, "streaming":self.key.stream,
                    "request_to_dispatch_ms":elapsed,
                    "timing_origin":"uni_api_handler_entry", "timing_end":"upstream_http_send"
                })
            );
            elapsed
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_from_arrival_and_counts_dispatch_once_before_completion() {
        let metrics = ChannelMetrics::new();
        let key = MetricKey::new("channel", "model", "upstream", "/v1/responses", true);
        let rows = vec![
            json!({"provider":"channel","model":"model","upstream_model":"upstream","endpoint":"/v1/responses","stream":true}),
        ];
        let arrival = RequestArrival(Instant::now() - std::time::Duration::from_millis(250));
        let dispatch = arrival.attempt(key.clone(), "request".into(), "attempt-1".into());
        // A selected plan that was never sent must not create a timing sample.
        let before = metrics.query(rows.clone(), "test", 15, false);
        assert_eq!(
            before["data"][0]["stats"]["request_to_dispatch"]["sample_count"],
            0
        );
        let first = dispatch.record(&metrics);
        assert!(first >= 250.);
        assert_eq!(dispatch.clone().record(&metrics), first);
        // Retrying uses the same arrival clock, not a new attempt clock.
        let retry = arrival.attempt(key.clone(), "request".into(), "attempt-2".into());
        let second = retry.record(&metrics);
        assert!(second >= first);
        metrics.observe_dispatch(&key, f64::NAN);
        metrics.observe_dispatch(&key, -1.);
        let after = metrics.query(rows, "test", 15, true);
        let stats = &after["data"][0]["stats"];
        assert_eq!(stats["request_to_dispatch"]["sample_count"], 2);
        assert_eq!(stats["request_to_dispatch"]["last_ms"], second);
        assert_eq!(stats["success_rate_denominator"], 0);
        assert_eq!(stats["started"], 0);
        assert_eq!(
            after["data"][0]["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["request_to_dispatch"]["sample_count"].as_u64().unwrap())
                .sum::<u64>(),
            2
        );
    }
}
