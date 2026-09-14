//! Volatile, bounded attempt metrics. Never reads/writes serving configuration.
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const RETENTION_MINUTES: u64 = 60;
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
    first_text: Distribution,
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
        self.first_text.merge(&other.first_text);
        self.duration.merge(&other.duration);
        self.last_success = self.last_success.max(other.last_success);
        self.last_failure = self.last_failure.max(other.last_failure);
    }
    fn json(&self) -> Value {
        let denominator = self.success + self.failed;
        json!({"started":self.started,"success":self.success,"failed":self.failed,"client_cancelled":self.client_cancelled,"hedge_cancelled":self.hedge_cancelled,"cancelled_unknown":self.cancelled_unknown,"skipped":self.skipped,"success_rate_denominator":denominator,"success_rate":(denominator>0).then(||self.success as f64/denominator as f64),"first_output":self.first_output.json(),"first_text":self.first_text.json(),"duration":self.duration.json(),"last_success_at":self.last_success,"last_failure_at":self.last_failure,"quality":if denominator == 0 {"no_samples"} else if denominator < 10 {"low_samples"} else {"sufficient"}})
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
    pub(crate) fn query(
        &self,
        entries: Vec<Value>,
        revision: &str,
        minutes: u64,
        timeseries: bool,
    ) -> Value {
        let now = self.now();
        let start_minute = (now / 60).saturating_sub(minutes - 1);
        let mut output = Vec::new();
        let (snapshots, dropped) = match self.inner.lock() {
            Ok(store) => (
                entries
                    .iter()
                    .map(|row| {
                        let key = MetricKey::new(
                            row["provider"].as_str().unwrap_or(""),
                            row["model"].as_str().unwrap_or(""),
                            row["upstream_model"].as_str().unwrap_or(""),
                            row["endpoint"].as_str().unwrap_or(""),
                            row["stream"].as_bool().unwrap_or(true),
                        );
                        let mut aggregate = Bucket::default();
                        let mut points = Vec::new();
                        let mut inflight = 0;
                        if let Some(series) = store.series.get(&key) {
                            inflight = series.inflight;
                            for bucket in &series.buckets {
                                if bucket.minute >= start_minute && bucket.minute <= now / 60 {
                                    aggregate.merge(bucket);
                                    if timeseries {
                                        points.push(bucket.clone());
                                    }
                                }
                            }
                        }
                        (aggregate, points, inflight)
                    })
                    .collect::<Vec<_>>(),
                store.dropped,
            ),
            Err(_) => (vec![(Bucket::default(), Vec::new(), 0); entries.len()], 1),
        };
        for (mut row, (aggregate, points, inflight)) in entries.into_iter().zip(snapshots) {
            row["stats"] = aggregate.json();
            row["stats"]["inflight"] = json!(inflight);
            if timeseries {
                row["points"] = Value::Array(
                    (start_minute..=now / 60)
                        .map(|minute| {
                            let mut point = points
                                .iter()
                                .find(|p| p.minute == minute)
                                .cloned()
                                .unwrap_or_default()
                                .json();
                            point["timestamp"] = json!(minute * 60);
                            point["covered"] = json!(minute * 60 >= self.started_at);
                            point
                        })
                        .collect(),
                );
            }
            output.push(row);
        }
        json!({"data":output,"scope":"instance","instance_id":self.instance_id.as_ref(),"collection_started_at":self.started_at,"generated_at":now,"from":start_minute*60,"to":now,"window_minutes":minutes,"bucket_seconds":60,"snapshot_revision":revision,"coverage":if self.started_at<=start_minute*60 && dropped==0 {"complete_for_instance"} else {"partial"},"dropped":dropped,"max_series":MAX_SERIES,"retention_seconds":3600,"measurement":"attempt_start_to_first_semantic_output","timing_sample_basis":"first_observed_at","success_sample_basis":"terminal_at","persistence":"memory"})
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
