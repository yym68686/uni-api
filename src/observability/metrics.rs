//! Volatile, bounded attempt metrics. Never reads/writes serving configuration.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Historical channel statistics are served by the S3/DuckDB analytics API.
// Keep only the current minute here for live gauges; this process must not
// become a second history store.
const RETENTION_MINUTES: u64 = 1;

const MAX_SERIES: usize = 1024;

const BOUNDS: [f64; 30] = [
    1.,
    2.,
    5.,
    10.,
    20.,
    50.,
    100.,
    200.,
    300.,
    500.,
    750.,
    1000.,
    1500.,
    2000.,
    3000.,
    5000.,
    7500.,
    10000.,
    15000.,
    20000.,
    30000.,
    45000.,
    60000.,
    90000.,
    120000.,
    180000.,
    300000.,
    600000.,
    1200000.,
    f64::INFINITY,
];

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MetricKey {
    pub provider: String,
    pub model: String,
    pub upstream_model: String,
    pub endpoint: String,
    pub stream: bool,
}

impl MetricKey {
    pub fn new(
        provider: &str,
        model: &str,
        upstream_model: &str,
        endpoint: &str,
        stream: bool,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            upstream_model: upstream_model.into(),
            endpoint: endpoint.into(),
            stream,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Distribution {
    bins: [u64; 30],
    count: u64,
    sum: f64,
    last: Option<(u64, f64)>,
}

impl Distribution {
    fn observe(&mut self, at: u64, value: f64) {
        if !value.is_finite() || value < 0. {
            return;
        }
        let bin = BOUNDS.iter().position(|b| value <= *b).unwrap_or(29);
        self.bins[bin] += 1;
        self.count += 1;
        self.sum += value;
        if self.last.is_none_or(|last| at >= last.0) {
            self.last = Some((at, value));
        }
    }
    fn merge(&mut self, other: &Self) {
        for (a, b) in self.bins.iter_mut().zip(other.bins) {
            *a += b;
        }
        self.count += other.count;
        self.sum += other.sum;
        if other
            .last
            .is_some_and(|v| self.last.is_none_or(|last| v.0 >= last.0))
        {
            self.last = other.last;
        }
    }
    fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let target = (self.count as f64 * q).ceil() as u64;
        let mut accumulated = 0;
        for (i, count) in self.bins.iter().enumerate() {
            accumulated += count;
            if accumulated >= target {
                // Upper bound estimate, never pretend the +Inf bucket has a finite bound.
                return BOUNDS[i].is_finite().then_some(BOUNDS[i]);
            }
        }
        None
    }
    fn json(&self) -> Value {
        json!({"sample_count":self.count,"mean_ms":(self.count>0).then(||self.sum/self.count as f64),"last_ms":self.last.map(|v|v.1),"last_observed_at":self.last.map(|v|v.0),"p50_ms":self.quantile(0.5),"p95_ms":self.quantile(0.95),"quantile_method":"histogram_upper_bound"})
    }
}

#[derive(Clone, Debug, Default)]
struct Bucket {
    minute: u64,
    started: u64,
    success: u64,
    failed: u64,
    client_cancelled: u64,
    hedge_cancelled: u64,
    cancelled_unknown: u64,
    skipped: u64,
    first_output: Distribution,
    request_to_dispatch: Distribution,
    first_text: Distribution,
    response_created: Distribution,
    duration: Distribution,
    last_success: Option<u64>,
    last_failure: Option<u64>,
}

impl Bucket {
    fn merge(&mut self, other: &Self) {
        self.started += other.started;
        self.success += other.success;
        self.failed += other.failed;
        self.client_cancelled += other.client_cancelled;
        self.hedge_cancelled += other.hedge_cancelled;
        self.cancelled_unknown += other.cancelled_unknown;
        self.skipped += other.skipped;
        self.first_output.merge(&other.first_output);
        self.request_to_dispatch.merge(&other.request_to_dispatch);
        self.first_text.merge(&other.first_text);
        self.response_created.merge(&other.response_created);
        self.duration.merge(&other.duration);
        self.last_success = self.last_success.max(other.last_success);
        self.last_failure = self.last_failure.max(other.last_failure);
    }
    fn json(&self) -> Value {
        let denominator = self.success + self.failed;
        json!({"started":self.started,"success":self.success,"failed":self.failed,"client_cancelled":self.client_cancelled,"hedge_cancelled":self.hedge_cancelled,"cancelled_unknown":self.cancelled_unknown,"skipped":self.skipped,"success_rate_denominator":denominator,"success_rate":(denominator>0).then(||self.success as f64/denominator as f64),"request_to_dispatch":self.request_to_dispatch.json(),"first_output":self.first_output.json(),"first_text":self.first_text.json(),"response_created":self.response_created.json(),"duration":self.duration.json(),"last_success_at":self.last_success,"last_failure_at":self.last_failure,"quality":if denominator == 0 {"no_samples"} else if denominator < 10 {"low_samples"} else {"sufficient"}})
    }
}

#[derive(Debug, Default)]
struct Series {
    buckets: VecDeque<Bucket>,
    inflight: u64,
}

impl Series {
    fn bucket(&mut self, minute: u64) -> &mut Bucket {
        while self
            .buckets
            .front()
            .is_some_and(|b| b.minute + RETENTION_MINUTES <= minute)
        {
            self.buckets.pop_front();
        }
        if self.buckets.back().is_none_or(|b| b.minute != minute) {
            self.buckets.push_back(Bucket {
                minute,
                ..Bucket::default()
            });
        }
        self.buckets.back_mut().expect("inserted current bucket")
    }
}

#[derive(Debug, Default)]
struct Store {
    series: HashMap<MetricKey, Series>,
    dropped: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ChannelMetrics {
    inner: Arc<Mutex<Store>>,
    epoch: Arc<Instant>,
    started_at: u64,
    instance_id: Arc<str>,
}

impl ChannelMetrics {
    pub(crate) fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            inner: Arc::new(Mutex::new(Store::default())),
            epoch: Arc::new(Instant::now()),
            started_at: now.as_secs(),
            instance_id: format!("{}-{:x}", std::process::id(), now.as_nanos()).into(),
        }
    }
    fn now(&self) -> u64 {
        self.started_at + self.epoch.elapsed().as_secs()
    }
    fn mutate(&self, key: &MetricKey, f: impl FnOnce(&mut Series, u64)) -> bool {
        let Ok(mut store) = self.inner.lock() else {
            return false;
        };
        let now = self.now();
        if !store.series.contains_key(key) && store.series.len() >= MAX_SERIES {
            store.series.retain(|_, s| {
                s.inflight > 0
                    || s.buckets
                        .back()
                        .is_some_and(|b| b.minute + RETENTION_MINUTES > now / 60)
            });
            if store.series.len() >= MAX_SERIES {
                store.dropped += 1;
                return false;
            }
        }
        f(store.series.entry(key.clone()).or_default(), now);
        true
    }
    pub(crate) fn start(
        &self,
        provider: &str,
        model: &str,
        upstream_model: &str,
        endpoint: &str,
        stream: bool,
    ) {
        let key = MetricKey::new(provider, model, upstream_model, endpoint, stream);
        self.mutate(&key, |s, now| {
            s.bucket(now / 60).started += 1;
            s.inflight += 1;
        });
    }

    pub(crate) fn observe_dispatch(&self, key: &MetricKey, elapsed_ms: f64) {
        if !elapsed_ms.is_finite() || elapsed_ms < 0. {
            return;
        }
        self.mutate(key, |series, now| {
            series
                .bucket(now / 60)
                .request_to_dispatch
                .observe(now, elapsed_ms);
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish(
        &self,
        provider: &str,
        model: &str,
        upstream_model: &str,
        endpoint: &str,
        stream: bool,
        outcome: &str,
        duration_ms: Option<f64>,
        first_output_ms: Option<f64>,
    ) {
        let key = MetricKey::new(provider, model, upstream_model, endpoint, stream);
        self.mutate(&key, |s, now| {
            s.inflight = s.inflight.saturating_sub(1);
            let b = s.bucket(now / 60);
            match outcome {
                "success" | "completed" | "incomplete" => {
                    b.success += 1;
                    b.last_success = Some(now)
                }
                "skipped" => b.skipped += 1,
                "cancelled" | "hedge_cancelled" => b.hedge_cancelled += 1,
                "client_cancelled" => b.client_cancelled += 1,
                _ => {
                    b.failed += 1;
                    b.last_failure = Some(now)
                }
            }
            if let Some(v) = duration_ms {
                b.duration.observe(now, v)
            }
            if let Some(v) = first_output_ms {
                b.first_output.observe(now, v)
            }
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn response_timings(
        &self,
        provider: &str,
        model: &str,
        upstream_model: &str,
        endpoint: &str,
        stream: bool,
        created_ms: Option<f64>,
        text_ms: Option<f64>,
    ) {
        let key = MetricKey::new(provider, model, upstream_model, endpoint, stream);
        self.mutate(&key, |series, now| {
            let bucket = series.bucket(now / 60);
            if let Some(value) = created_ms {
                bucket.response_created.observe(now, value);
            }
            if let Some(value) = text_ms {
                bucket.first_text.observe(now, value);
            }
        });
    }
    pub(crate) fn query(
        &self,
        entries: Vec<Value>,
        revision: &str,
        minutes: u64,
        timeseries: bool,
    ) -> Value {
        let now = self.now();
        let effective_minutes = minutes.min(1);
        let start_minute = (now / 60).saturating_sub(effective_minutes - 1);
        let mut output = Vec::new();
        let (snapshots, dropped, endpoints) = match self.inner.lock() {
            Ok(store) => (
                {
                    // Visit each stored series once. Aggregate rows match either endpoint
                    // and/or stream dimension; histograms merge before quantiles are computed.
                    let mut snapshots =
                        vec![(Bucket::default(), BTreeMap::<u64, Bucket>::new(), 0); entries.len()];
                    let mut targets = HashMap::<(&str, &str, &str), Vec<usize>>::new();
                    for (i, row) in entries.iter().enumerate() {
                        targets
                            .entry((
                                row["provider"].as_str().unwrap_or(""),
                                row["model"].as_str().unwrap_or(""),
                                row["upstream_model"].as_str().unwrap_or(""),
                            ))
                            .or_default()
                            .push(i);
                    }
                    for (source_key, series) in &store.series {
                        let Some(indices) = targets.get(&(
                            source_key.provider.as_str(),
                            source_key.model.as_str(),
                            source_key.upstream_model.as_str(),
                        )) else {
                            continue;
                        };
                        for &i in indices {
                            let row = &entries[i];
                            let endpoint_matches = row["endpoint"].as_str() == Some("all")
                                || row["endpoint"].as_str() == Some(source_key.endpoint.as_str());
                            let stream_matches = row["stream"].is_null()
                                || row["stream"].as_bool() == Some(source_key.stream);
                            if !endpoint_matches || !stream_matches {
                                continue;
                            }
                            let (aggregate, points, inflight) = &mut snapshots[i];
                            *inflight += series.inflight;
                            for bucket in &series.buckets {
                                if bucket.minute >= start_minute && bucket.minute <= now / 60 {
                                    aggregate.merge(bucket);
                                    if timeseries {
                                        points.entry(bucket.minute).or_default().merge(bucket);
                                    }
                                }
                            }
                        }
                    }
                    snapshots
                },
                store.dropped,
                store
                    .series
                    .keys()
                    .map(|key| key.endpoint.clone())
                    .collect::<BTreeSet<_>>(),
            ),
            Err(_) => (
                vec![(Bucket::default(), BTreeMap::new(), 0); entries.len()],
                1,
                BTreeSet::new(),
            ),
        };
        for (mut row, (aggregate, points, inflight)) in entries.into_iter().zip(snapshots) {
            row["stats"] = aggregate.json();
            row["stats"]["inflight"] = json!(inflight);
            if timeseries {
                row["points"] = Value::Array(
                    (start_minute..=now / 60)
                        .map(|minute| {
                            let mut point = points.get(&minute).cloned().unwrap_or_default().json();
                            point["timestamp"] = json!(minute * 60);
                            point["covered"] = json!(minute * 60 >= self.started_at);
                            point
                        })
                        .collect(),
                );
            }
            output.push(row);
        }
        json!({"data":output,"available_endpoints":endpoints,"scope":"instance","instance_id":self.instance_id.as_ref(),"collection_started_at":self.started_at,"generated_at":now,"from":start_minute*60,"to":now,"window_minutes":effective_minutes,"bucket_seconds":60,"snapshot_revision":revision,"coverage":"live_only","dropped":dropped,"max_series":MAX_SERIES,"retention_seconds":60,"measurement":"live_attempt_gauge","timing_sample_basis":"first_observed_at","success_sample_basis":"terminal_at","request_to_dispatch_measurement":"uni_api_handler_entry_to_upstream_http_send","request_to_dispatch_sample_basis":"dispatch_at","persistence":"live_gauge"})
    }
}

pub(crate) fn global() -> ChannelMetrics {
    static INSTANCE: OnceLock<ChannelMetrics> = OnceLock::new();
    INSTANCE.get_or_init(ChannelMetrics::new).clone()
}

pub(crate) fn parse_window(raw: Option<&str>, default: std::time::Duration) -> std::time::Duration {
    let value = raw.unwrap_or_default().trim().to_ascii_lowercase();
    let (number, multiplier) = if let Some(v) = value.strip_suffix('m') {
        (v, 60)
    } else if let Some(v) = value.strip_suffix('h') {
        (v, 3600)
    } else {
        return default;
    };
    number
        .parse::<u64>()
        .ok()
        .filter(|v| *v > 0)
        .map(|v| {
            std::time::Duration::from_secs(v.saturating_mul(multiplier).min(RETENTION_MINUTES * 60))
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_endpoint_and_stream_filters_merge_histograms_without_double_counting() {
        let metrics = ChannelMetrics::new();
        // A large fast population and one slow sample must not average p50 values.
        for (endpoint, stream, count, elapsed, outcome) in [
            ("/v1/messages", true, 19, 100., "completed"),
            ("/v1/messages", false, 1, 5000., "failed"),
            ("/v1/responses", true, 2, 200., "completed"),
            ("/v1/responses", false, 3, 300., "completed"),
        ] {
            for _ in 0..count {
                metrics.start("p", "m", "u", endpoint, stream);
                metrics.observe_dispatch(&MetricKey::new("p", "m", "u", endpoint, stream), elapsed);
                metrics.finish(
                    "p",
                    "m",
                    "u",
                    endpoint,
                    stream,
                    outcome,
                    Some(elapsed),
                    Some(elapsed),
                );
            }
        }
        metrics.start("other-provider", "m", "u", "/v1/messages", true);
        metrics.start("p", "other-model", "u", "/v1/messages", true);
        metrics.start("p", "m", "old-upstream", "/v1/messages", true);
        for (endpoint, stream, started, success, failed) in [
            ("all", Value::Null, 25, 24, 1),
            ("all", json!(true), 21, 21, 0),
            ("all", json!(false), 4, 3, 1),
            ("/v1/messages", Value::Null, 20, 19, 1),
            ("/v1/messages", json!(true), 19, 19, 0),
            ("/v1/messages", json!(false), 1, 0, 1),
            ("/v1/chat/completions", Value::Null, 0, 0, 0),
        ] {
            let output = metrics.query(vec![json!({"provider":"p","model":"m","upstream_model":"u","endpoint":endpoint,"stream":stream})], "test", 60, true);
            let row = &output["data"][0];
            let stats = &row["stats"];
            assert_eq!(stats["started"], started);
            assert_eq!(stats["success"], success);
            assert_eq!(stats["failed"], failed);
            assert_eq!(stats["request_to_dispatch"]["sample_count"], started);
            assert_eq!(stats["inflight"], 0);
            assert_eq!(
                row["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["started"].as_u64().unwrap())
                    .sum::<u64>(),
                started
            );
            if endpoint == "all" && stream.is_null() {
                assert_eq!(stats["first_output"]["p50_ms"], 100.);
                assert_eq!(stats["first_output"]["p95_ms"], 300.);
                assert_eq!(stats["success_rate"], 24. / 25.);
            }
        }
    }
}
