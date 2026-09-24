use crate::config::snapshot::Provider;
use crate::config::snapshot::Snapshot;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::time::Duration;

/// Every transport and protocol adapter uses the same disabled-value semantics.
/// In particular, an explicit zero must never acquire a fallback deadline.
pub(crate) fn positive_duration(seconds: Option<f64>) -> Option<Duration> {
    seconds
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
        .map(|duration| duration.max(Duration::from_nanos(1)))
}

pub(crate) fn earliest_timeout(values: &[Option<f64>]) -> Option<Duration> {
    values.iter().copied().filter_map(positive_duration).min()
}

/// Preserve disabled timeouts when crossing a protocol adapter. Only an enabled
/// deadline that has actually expired becomes an immediate deadline.
pub(crate) fn remaining_timeout(seconds: Option<f64>, elapsed: Duration) -> Option<f64> {
    positive_duration(seconds).map(|duration| {
        duration
            .saturating_sub(elapsed)
            .max(Duration::from_nanos(1))
            .as_secs_f64()
    })
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Timeouts {
    pub(crate) connect: Option<f64>,
    pub(crate) write: Option<f64>,
    pub(crate) pool: Option<f64>,
    pub(crate) first_byte: Option<f64>,
    pub(crate) idle: Option<f64>,
    pub(crate) total: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_timeouts(
    snapshot: &Snapshot,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    engine: &str,
    stream: bool,
    request_type: Option<&str>,
    role: &str,
    endpoint: &str,
    method: &str,
) -> Timeouts {
    let base = model_timeout(
        provider,
        &snapshot.preferences,
        request_model,
        original_model,
    );
    let context = HashMap::from([
        ("provider", provider.name.as_ref()),
        ("endpoint", endpoint),
        ("method", method),
        ("engine", engine),
        ("model", request_model),
        ("request_model", request_model),
        ("upstream_model", original_model),
        ("request_type", request_type.unwrap_or_default()),
        ("role", role),
    ]);
    let mut values = Map::new();
    merge_timeout_policy(
        &mut values,
        snapshot.preferences.get("timeout_policy"),
        &context,
        stream,
    );
    merge_timeout_policy(
        &mut values,
        provider.preferences.get("timeout_policy"),
        &context,
        stream,
    );
    Timeouts {
        connect: values.get("connect").and_then(Value::as_f64),
        write: values.get("write").and_then(Value::as_f64),
        pool: values.get("pool").and_then(Value::as_f64),
        first_byte: values
            .get("first_byte")
            .and_then(Value::as_f64)
            // Non-streaming providers may withhold headers until generation is
            // complete. Preserve the legacy total-only policy instead of
            // silently shortening it with the model's streaming fallback.
            // An explicitly configured first_byte (including zero) still wins.
            .or_else(|| {
                if stream {
                    None
                } else {
                    values.get("total").and_then(Value::as_f64)
                }
            })
            .or(Some(base)),
        idle: values.get("idle").and_then(Value::as_f64),
        total: values.get("total").and_then(Value::as_f64),
    }
}

pub(crate) fn model_timeout(
    provider: &Provider,
    global: &Map<String, Value>,
    request_model: &str,
    original_model: &str,
) -> f64 {
    model_preference(
        provider,
        global,
        request_model,
        original_model,
        "model_timeout",
    )
    .unwrap_or(100.0)
}

pub(crate) fn model_preference(
    provider: &Provider,
    global: &Map<String, Value>,
    request_model: &str,
    original_model: &str,
    preference: &str,
) -> Option<f64> {
    for preferences in [&provider.preferences, global] {
        let Some(timeout) = preferences.get(preference) else {
            continue;
        };
        if let Some(value) = timeout.as_f64() {
            return Some(value);
        }
        let Some(values) = timeout.as_object() else {
            continue;
        };
        if let Some(value) = model_timeout_value(values, request_model) {
            return Some(value);
        }
        if let Some(value) = model_timeout_value(values, original_model) {
            return Some(value);
        }
        if let Some(value) = values
            .iter()
            .find(|(key, value)| key.eq_ignore_ascii_case("default") && value.as_f64().is_some())
            .and_then(|(_, value)| value.as_f64())
        {
            return Some(value);
        }
    }
    None
}

pub(crate) fn model_timeout_value(values: &Map<String, Value>, model: &str) -> Option<f64> {
    let normalized_model = model.to_ascii_lowercase();

    values
        .iter()
        .find(|(key, value)| {
            !key.eq_ignore_ascii_case("default")
                && key.eq_ignore_ascii_case(model)
                && value.as_f64().is_some()
        })
        .and_then(|(_, value)| value.as_f64())
        .or_else(|| {
            values
                .iter()
                .find(|(key, value)| {
                    !key.is_empty()
                        && !key.eq_ignore_ascii_case("default")
                        && normalized_model.contains(&key.to_ascii_lowercase())
                        && value.as_f64().is_some()
                })
                .and_then(|(_, value)| value.as_f64())
        })
}

pub(crate) fn merge_timeout_policy(
    target: &mut Map<String, Value>,
    policy: Option<&Value>,
    context: &HashMap<&str, &str>,
    stream: bool,
) {
    let Some(policy) = policy.and_then(Value::as_object) else {
        return;
    };
    if let Some(default) = policy.get("default").and_then(Value::as_object) {
        target.extend(default.clone());
    }
    let mut best: Option<(&Map<String, Value>, usize)> = None;
    for rule in policy
        .get("rules")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let Some(condition) = rule.get("match").and_then(Value::as_object) else {
            continue;
        };
        if timeout_rule_matches(condition, context, stream)
            && best.is_none_or(|(_, score)| condition.len() > score)
        {
            if let Some(timeout) = rule.get("timeout").and_then(Value::as_object) {
                best = Some((timeout, condition.len()));
            }
        }
    }
    if let Some((timeout, _)) = best {
        target.extend(timeout.clone());
    }
}

pub(crate) fn timeout_rule_matches(
    condition: &Map<String, Value>,
    context: &HashMap<&str, &str>,
    stream: bool,
) -> bool {
    condition.iter().all(|(key, expected)| {
        if key == "stream" {
            return expected.as_bool() == Some(stream);
        }
        let actual = context.get(key.as_str()).copied().unwrap_or_default();
        timeout_value_matches(expected, actual)
    })
}

pub(crate) fn timeout_value_matches(expected: &Value, actual: &str) -> bool {
    if let Some(values) = expected.as_array() {
        return values
            .iter()
            .any(|value| timeout_value_matches(value, actual));
    }
    let Some(expected) = expected.as_str() else {
        return false;
    };
    expected == "*"
        || expected.eq_ignore_ascii_case(actual)
        || expected.strip_suffix('*').is_some_and(|prefix| {
            actual
                .to_ascii_lowercase()
                .starts_with(&prefix.to_ascii_lowercase())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_and_invalid_values_cannot_turn_into_deadlines() {
        for value in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::INFINITY),
            Some(f64::NAN),
            Some(f64::MAX),
        ] {
            assert_eq!(positive_duration(value), None);
            assert_eq!(remaining_timeout(value, Duration::from_secs(1)), None);
        }
        assert_eq!(
            earliest_timeout(&[Some(0.0), None, Some(2.0), Some(1.0)]),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            remaining_timeout(Some(2.0), Duration::from_secs(1)),
            Some(1.0)
        );
        assert_eq!(
            remaining_timeout(Some(2.0), Duration::from_secs(3)),
            Some(1e-9)
        );
    }
}
