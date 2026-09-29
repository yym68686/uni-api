//! Global preference intent shares the encrypted console settings journal and
//! atomic restore contract. The reserved scope is never a routable provider.
use super::settings::{merge, ProviderSettings};
use serde_json::{json, Map, Value};

pub(crate) const SCOPE: &str = "__uni_console_global_preferences__";
pub(crate) const KEYS: &[&str] = &[
    "model_timeout",
    "timeout_policy",
    "keepalive_interval",
    "cooldown_period",
    "api_key_cooldown_period",
    "api_key_rate_limit_cooldown_period",
    "api_key_quota_cooldown_period",
    "hedging",
];

pub(crate) fn document(preferences: &Map<String, Value>) -> Value {
    json!({"preferences":preferences.iter().filter(|(k,_)|KEYS.contains(&k.as_str())).map(|(k,v)|(k.clone(),v.clone())).collect::<Map<_,_>>()})
}
pub(crate) fn schema() -> Value {
    json!({"version":1,"global_settings":true,"fields":KEYS.iter().map(|k|json!({"path":format!("/preferences/{k}"),"group":if *k=="hedging"{"并行请求"}else{"超时与冷却"},"type":if k.ends_with("cooldown_period"){"number"}else{"json"}})).collect::<Vec<_>>(),"engines":[],"key_algorithms":[]})
}
pub(crate) fn default_value(key: &str) -> Option<Value> {
    Some(match key {
        "model_timeout" => json!(100),
        "timeout_policy" => json!({"default":{},"rules":[]}),
        "keepalive_interval" => json!(99999),
        "cooldown_period" | "api_key_cooldown_period" => json!(0),
        "api_key_rate_limit_cooldown_period" => json!(1800),
        "api_key_quota_cooldown_period" => json!(21600),
        "hedging" => {
            json!({"enabled":false,"max_inflight_attempts":1,"winner_policy":"first_valid_success"})
        }
        _ => return None,
    })
}
pub(crate) fn validate(raw: &Value) -> Result<(), String> {
    let p = raw
        .get("preferences")
        .and_then(Value::as_object)
        .ok_or("Invalid global preferences")?;
    let numeric = |v: &Value| v.as_f64().is_some_and(|n| n >= 0.0 && n.is_finite());
    for (k, v) in p {
        if !KEYS.contains(&k.as_str()) {
            return Err(format!("Unsupported global preference: {k}"));
        }
        if k.ends_with("cooldown_period") && !numeric(v) {
            return Err(format!("Invalid global preference: {k}"));
        }
        if ["model_timeout", "keepalive_interval"].contains(&k.as_str())
            && !(numeric(v) || v.as_object().is_some_and(|m| m.values().all(numeric)))
        {
            return Err(format!("Invalid global preference: {k}"));
        }
        if k == "timeout_policy" {
            validate_timeout_policy(v)?;
        }
        if k == "hedging" {
            let h = v.as_object().ok_or("Invalid hedging settings")?;
            if h.keys().any(|k| {
                !["enabled", "max_inflight_attempts", "winner_policy"].contains(&k.as_str())
            }) || h.get("enabled").is_some_and(|v| !v.is_boolean())
                || h.get("max_inflight_attempts")
                    .is_some_and(|v| !v.as_u64().is_some_and(|n| (1..=4).contains(&n)))
                || h.get("winner_policy")
                    .is_some_and(|v| v.as_str() != Some("first_valid_success"))
            {
                return Err("Invalid hedging settings".into());
            }
        }
    }
    Ok(())
}
pub(crate) fn validate_timeout_policy(v: &Value) -> Result<(), String> {
    let timeout = |v: &Value| {
        v.as_object().is_some_and(|o| {
            o.iter().all(|(k, v)| {
                ["connect", "write", "pool", "first_byte", "idle", "total"].contains(&k.as_str())
                    && v.as_f64().is_some_and(|n| n >= 0.0 && n.is_finite())
            })
        })
    };
    let condition = |v: &Value| {
        v.as_object().is_some_and(|o| {
            o.iter().all(|(k, v)| {
                if k == "stream" {
                    return v.is_boolean();
                }
                let string = |v: &Value| v.as_str().is_some_and(|s| !s.trim().is_empty());
                [
                    "provider",
                    "endpoint",
                    "method",
                    "engine",
                    "model",
                    "request_model",
                    "upstream_model",
                    "request_type",
                    "role",
                ]
                .contains(&k.as_str())
                    && (string(v)
                        || v.as_array()
                            .is_some_and(|a| !a.is_empty() && a.iter().all(string)))
            })
        })
    };
    if !v
        .as_object()
        .is_some_and(|o| o.keys().all(|k| ["default", "rules"].contains(&k.as_str())))
        || v.get("default").is_some_and(|v| !timeout(v))
        || v.get("rules").is_some_and(|v| {
            !v.as_array().is_some_and(|r| {
                r.iter().all(|r| {
                    r.as_object().is_some_and(|o| {
                        o.keys().all(|k| ["match", "timeout"].contains(&k.as_str()))
                    }) && r.get("match").is_some_and(condition)
                        && r.get("timeout").is_some_and(timeout)
                })
            })
        })
    {
        return Err("Invalid timeout policy".into());
    }
    Ok(())
}
pub(crate) fn validate_intent(settings: &ProviderSettings) -> Result<(), String> {
    for path in settings.set.keys().chain(settings.remove.iter()) {
        if !KEYS.iter().any(|k| {
            path == &format!("/preferences/{k}") || path.starts_with(&format!("/preferences/{k}/"))
        }) {
            return Err("Unsupported global settings path".into());
        }
    }
    validate(&merge(
        &json!({"preferences":{}}),
        &settings.set,
        &settings.remove,
    ))
}
pub(crate) fn overlay(
    base: &Map<String, Value>,
    settings: Option<&ProviderSettings>,
) -> Map<String, Value> {
    let Some(s) = settings else {
        return base.clone();
    };
    let merged = merge(&json!({"preferences":base}), &s.set, &s.remove);
    // Mutations and restored intent are validated before entering the overlay.
    merged["preferences"]
        .as_object()
        .cloned()
        .unwrap_or_else(|| base.clone())
}

pub(crate) fn note(key: &str) -> &'static str {
    match key {
        "model_timeout" => "支持统一秒数或按模型设置；条件超时策略优先，具体请求值可在预览中核对。",
        "timeout_policy" => "全局规则先匹配，渠道规则再覆盖；未填写的超时项继续继承。",
        "keepalive_interval" => "0 或大于当前模型超时的间隔不发送心跳。",
        "api_key_rate_limit_cooldown_period" => {
            "配置为 0 时实际使用 1800 秒；站点 Retry-After 可延长冷却。"
        }
        "api_key_quota_cooldown_period" => "配置为 0 时实际使用 21600 秒。",
        _ => "",
    }
}

pub(crate) fn field_values(channel: &Value, global: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for k in KEYS.iter().filter(|k| **k != "hedging") {
        let path = format!("/preferences/{k}");
        let configured = channel.pointer(&path);
        let inherited = global
            .get(*k)
            .cloned()
            .or_else(|| default_value(k))
            .unwrap_or(Value::Null);
        let value = configured.cloned().unwrap_or_else(|| inherited.clone());
        out.insert(path,json!({"value":value,"source":if configured.is_some(){"channel"}else if global.contains_key(*k){"global"}else{"default"},"inherited_value":inherited,"inherited_source":if global.contains_key(*k){"global"}else{"default"},"can_inherit":true,"note":note(k)}));
    }
    out.insert("/AUTO_RETRY".into(),json!({"value":"由调用 API key 决定（未设置时开启）","source":"api_key","can_inherit":false,"read_only":true,"note":"本运行时的重试策略由调用 API key 控制，渠道同名字段不会改变请求重试。"}));
    Value::Object(out)
}

pub(crate) fn resolved_hedging(p: &Map<String, Value>) -> Value {
    let h = crate::upstream::hedging::parse_hedging(p);
    json!({"enabled":h.enabled,"max_inflight_attempts":h.max_inflight_attempts,"winner_policy":"first_valid_success","active":h.active()})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_validation_and_cooldown_resolution_match_runtime() {
        for policy in [
            json!({"rules":[{"match":{"stream":"false"},"timeout":{"total":10}}]}),
            json!({"rules":[{"match":{"endpoint":[]},"timeout":{"total":10}}]}),
            json!({"default":{"unknown":10}}),
            json!({"default":{"total":-1}}),
        ] {
            assert!(validate_timeout_policy(&policy).is_err());
        }
        assert!(validate_timeout_policy(&json!({"rules":[{"match":{"model":["gpt-5*","gpt-6*"],"stream":false},"timeout":{"total":0}}]})).is_ok());
        let globals =
            json!({"api_key_rate_limit_cooldown_period":90,"api_key_quota_cooldown_period":120});
        let channel = json!({"api_key_rate_limit_cooldown_period":0});
        let seconds = |key| {
            crate::runtime::scheduling::cooldown_seconds(
                channel.as_object().unwrap(),
                globals.as_object().unwrap(),
                key,
            )
        };
        assert_eq!(seconds("api_key_rate_limit_cooldown_period"), 1800.0);
        assert_eq!(seconds("api_key_quota_cooldown_period"), 120.0);
        assert_eq!(seconds("api_key_cooldown_period"), 0.0);
    }
}
