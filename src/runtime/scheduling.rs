use crate::config::snapshot::Provider;
use crate::routing::access::preference_f64;
use crate::routing::planner::scheduling_seed;
use crate::routing::planner::shuffle_indices;
use crate::routing::types::FailedRoute;
use crate::runtime::state::GatewayRuntime;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub(crate) type RouteKey = (String, String);

pub(crate) type RouteFailureHistory = HashMap<RouteKey, VecDeque<tokio::time::Instant>>;

pub(crate) type RateWindows = HashMap<(String, u64), VecDeque<tokio::time::Instant>>;

pub(crate) enum ProviderKeySelection {
    Selected(String),
    NoProviderKey,
    ChannelCooling,
    AllKeysCooling,
}

impl GatewayRuntime {
    pub(crate) async fn select_provider_key(
        &self,
        provider: &Provider,
        original_model: &str,
    ) -> ProviderKeySelection {
        if provider.api_keys.is_empty()
            && provider.client_email.is_some()
            && provider.private_key.is_some()
        {
            return ProviderKeySelection::Selected(String::new());
        }
        if provider.api_keys.is_empty() {
            return ProviderKeySelection::NoProviderKey;
        }
        let now = tokio::time::Instant::now();
        if self
            .scheduling
            .channel_cooldowns
            .lock()
            .await
            .get(&(provider.name.to_string(), original_model.to_owned()))
            .is_some_and(|until| *until > now)
        {
            return ProviderKeySelection::ChannelCooling;
        }
        let algorithm = provider
            .preferences
            .get("api_key_schedule_algorithm")
            .or_else(|| provider.preferences.get("API_KEY_SCHEDULE_ALGORITHM"))
            .and_then(Value::as_str)
            .unwrap_or("round_robin")
            .trim()
            .to_ascii_lowercase();
        let cooldowns = self.scheduling.key_cooldowns.lock().await;
        let mut candidates = (0..provider.api_keys.len()).collect::<Vec<_>>();
        if algorithm == "fixed_priority" || algorithm == "priority" {
            // preserve configured order
        } else if algorithm == "random" || algorithm == "lottery" {
            shuffle_indices(
                &mut candidates,
                scheduling_seed(provider.name.as_ref(), original_model),
            );
        } else {
            let start = provider.cursor.fetch_add(1, Ordering::Relaxed) % provider.api_keys.len();
            candidates.rotate_left(start);
        }
        drop(cooldowns);
        for index in candidates {
            let key = provider.api_keys[index].clone();
            let cooling = self
                .scheduling
                .key_cooldowns
                .lock()
                .await
                .get(&(provider.name.to_string(), key.clone()))
                .is_some_and(|until| *until > now);
            if cooling {
                continue;
            }
            if let Some(rules) = parse_rate_limits(
                provider.preferences.get("api_key_rate_limit"),
                Some(original_model),
            ) {
                if !self.admit_provider_rate(provider, &key, &rules).await {
                    continue;
                }
            }
            return ProviderKeySelection::Selected(key);
        }
        ProviderKeySelection::AllKeysCooling
    }

    pub(crate) async fn cool_failed_route(&self, failure: FailedRoute<'_>) {
        let FailedRoute {
            provider,
            key,
            original_model,
            has_alternative,
            status,
            detail,
            provider_model_unavailable,
            force_quota_cooldown,
        } = failure;
        if provider_model_unavailable || matches!(status, 403 | 404) {
            let now = tokio::time::Instant::now();
            let route_key = (provider.name.to_string(), original_model.to_owned());
            let mut failures = self.scheduling.route_failures.lock().await;
            let history = failures.entry(route_key.clone()).or_default();
            while history.front().is_some_and(|observed| {
                now.duration_since(*observed) >= provider_model_circuit_window()
            }) {
                history.pop_front();
            }
            history.push_back(now);
            if history.len() >= provider_model_circuit_threshold() {
                self.scheduling
                    .channel_cooldowns
                    .lock()
                    .await
                    .insert(route_key, now + provider_model_circuit_open_period());
            }
        }
        let global_seconds = self
            .current
            .read()
            .await
            .as_ref()
            .and_then(|snapshot| preference_f64(&snapshot.preferences, "cooldown_period"));
        let channel_seconds = preference_f64(&provider.preferences, "cooldown_period")
            .or(global_seconds)
            .unwrap_or(0.0);
        if has_alternative && channel_seconds > 0.0 {
            self.scheduling.channel_cooldowns.lock().await.insert(
                (provider.name.to_string(), original_model.to_owned()),
                tokio::time::Instant::now() + Duration::from_secs_f64(channel_seconds),
            );
        }
        if provider_model_unavailable || provider.api_keys.len() <= 1 {
            return;
        }
        let lower_detail = detail.to_ascii_lowercase();
        let quota_failure = force_quota_cooldown
            || matches!(status, 401..=403) && provider.engine.eq_ignore_ascii_case("codex")
            || lower_detail.contains("insufficient_quota")
            || lower_detail.contains("billing_hard_limit_reached");
        let key_seconds = if quota_failure {
            preference_f64(&provider.preferences, "api_key_quota_cooldown_period")
                .filter(|value| *value > 0.0)
                .unwrap_or(6.0 * 60.0 * 60.0)
        } else if status == 429
            && [
                "rate_limit_exceeded",
                "rate limit reached",
                "too many requests",
                "tokens per min",
                "requests per min",
                "tokens per day",
                "requests per day",
                "please try again in",
            ]
            .iter()
            .any(|marker| lower_detail.contains(marker))
        {
            preference_f64(&provider.preferences, "api_key_rate_limit_cooldown_period")
                .filter(|value| *value > 0.0)
                .unwrap_or(30.0 * 60.0)
                .max(retry_after_seconds(detail).unwrap_or(0.0))
        } else {
            preference_f64(&provider.preferences, "api_key_cooldown_period").unwrap_or(0.0)
        };
        if key_seconds > 0.0 {
            self.scheduling.key_cooldowns.lock().await.insert(
                (provider.name.to_string(), key.to_owned()),
                tokio::time::Instant::now() + Duration::from_secs_f64(key_seconds),
            );
        }
    }

    async fn admit_provider_rate(
        &self,
        provider: &Provider,
        key: &str,
        rules: &[(usize, u64)],
    ) -> bool {
        let now = tokio::time::Instant::now();
        let mut buckets = self.scheduling.provider_windows.lock().await;
        for (limit, seconds) in rules {
            if *seconds == 0 {
                continue;
            }
            let bucket = format!("provider:{}:{}", provider.name, key);
            let queue = buckets.entry((bucket, *seconds)).or_default();
            let window = Duration::from_secs(*seconds);
            while queue
                .front()
                .is_some_and(|started| now.duration_since(*started) >= window)
            {
                queue.pop_front();
            }
            if queue.len() >= *limit {
                return false;
            }
        }
        for (_, seconds) in rules {
            if *seconds > 0 {
                buckets
                    .entry((format!("provider:{}:{}", provider.name, key), *seconds))
                    .or_default()
                    .push_back(now);
            }
        }
        true
    }

    pub(crate) async fn reset_route_failure(&self, provider: &Provider, original_model: &str) {
        self.scheduling
            .route_failures
            .lock()
            .await
            .remove(&(provider.name.to_string(), original_model.to_owned()));
    }

    pub(crate) async fn admit_rate(&self, bucket: &str, rules: &[(usize, u64)]) -> bool {
        let now = tokio::time::Instant::now();
        let mut buckets = self.scheduling.client_windows.lock().await;
        for (limit, seconds) in rules {
            if *seconds == 0 {
                continue;
            }
            let queue = buckets.entry((bucket.to_owned(), *seconds)).or_default();
            let window = Duration::from_secs(*seconds);
            while queue
                .front()
                .is_some_and(|started| now.duration_since(*started) >= window)
            {
                queue.pop_front();
            }
            if queue.len() >= *limit {
                return false;
            }
        }
        for (_, seconds) in rules {
            if *seconds == 0 {
                continue;
            }
            buckets
                .entry((bucket.to_owned(), *seconds))
                .or_default()
                .push_back(now);
        }
        true
    }
}

pub(crate) fn provider_model_circuit_threshold() -> usize {
    std::env::var("PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3)
}

pub(crate) fn provider_model_circuit_window() -> Duration {
    environment_duration("PROVIDER_MODEL_CIRCUIT_WINDOW_SECONDS", 120.0)
}

pub(crate) fn provider_model_circuit_open_period() -> Duration {
    environment_duration("PROVIDER_MODEL_CIRCUIT_OPEN_SECONDS", 300.0)
}

pub(crate) fn environment_duration(name: &str, default_seconds: f64) -> Duration {
    Duration::from_secs_f64(
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(default_seconds),
    )
}

pub(crate) fn retry_after_seconds(detail: &str) -> Option<f64> {
    let lower = detail.to_ascii_lowercase();
    let tail = lower.split_once("try again in")?.1.trim_start();
    let number_end = tail
        .find(|character: char| !(character.is_ascii_digit() || character == '.'))
        .unwrap_or(tail.len());
    if number_end == 0 {
        return None;
    }
    let number = tail[..number_end].parse::<f64>().ok()?;
    let unit = tail[number_end..].trim_start();
    Some(
        if unit.starts_with("ms") || unit.starts_with("millisecond") {
            (number / 1000.0).ceil()
        } else if unit.starts_with('m') {
            (number * 60.0).ceil()
        } else {
            number.ceil()
        },
    )
}

pub(crate) fn tpr_exceeded(rules: &[(usize, u64)], estimated_tokens: usize) -> bool {
    rules
        .iter()
        .any(|(limit, seconds)| *seconds == 0 && estimated_tokens > *limit)
}

pub(crate) fn parse_rate_limits(
    value: Option<&Value>,
    model: Option<&str>,
) -> Option<Vec<(usize, u64)>> {
    let raw = match value {
        None | Some(Value::Null) => "999999/min",
        Some(Value::String(value)) => value.trim(),
        Some(Value::Object(values)) => {
            let selected = if let Some(exact) = model.and_then(|model| values.get(model)) {
                Some(exact)
            } else {
                let matches = model
                    .into_iter()
                    .flat_map(|model| {
                        values.iter().filter(move |(configured, _)| {
                            configured.as_str() != "default" && model.contains(configured.as_str())
                        })
                    })
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>();
                if matches.len() > 1 {
                    return None;
                }
                matches.first().copied().or_else(|| values.get("default"))
            };
            match selected {
                Some(Value::String(value)) => value.trim(),
                Some(_) => return None,
                None => "999999/min",
            }
        }
        _ => return None,
    };
    let mut rules = Vec::new();
    for configured in raw.split(',') {
        let (count, period) = configured.trim().split_once('/')?;
        let count = count.trim().parse::<usize>().ok()?;
        let seconds = match period.trim().to_ascii_lowercase().as_str() {
            "s" | "sec" | "second" => 1,
            "m" | "min" | "minute" => 60,
            "h" | "hr" | "hour" => 3_600,
            "d" | "day" => 86_400,
            "mo" | "month" => 2_592_000,
            "y" | "year" => 31_536_000,
            "tpr" => 0,
            _ => return None,
        };
        rules.push((count, seconds));
    }
    Some(rules)
}

#[derive(Clone, Default)]
pub(crate) struct SchedulingState {
    pub(crate) provider_cursors: Arc<Mutex<HashMap<String, Arc<AtomicUsize>>>>,
    pub(crate) key_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    pub(crate) channel_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    pub(crate) route_failures: Arc<Mutex<RouteFailureHistory>>,
    pub(crate) client_windows: Arc<Mutex<RateWindows>>,
    pub(crate) provider_windows: Arc<Mutex<RateWindows>>,
    pub(crate) routing_cursors: Arc<Mutex<HashMap<(String, String), usize>>>,
}
