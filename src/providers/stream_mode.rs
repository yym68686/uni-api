//! Transport capability for Responses upstreams, independent of the caller's wire format.
use crate::config::snapshot::Provider;
use serde_json::Value;

pub(crate) fn validate(value: &Value) -> Result<(), String> {
    let mode = |v: &Value| matches!(v.as_str(), Some("follow_client" | "stream_only"));
    let valid = mode(value)
        || value.as_object().is_some_and(|m| {
            m.keys().all(|k| matches!(k.as_str(), "default" | "models"))
                && m.get("default").is_none_or(mode)
                && m.get("models").is_none_or(|v| {
                    v.as_object().is_some_and(|models| {
                        models
                            .iter()
                            .all(|(name, value)| !name.trim().is_empty() && mode(value))
                    })
                })
        });
    if valid {
        Ok(())
    } else {
        Err("Invalid channel setting: /preferences/upstream_stream_mode".into())
    }
}

pub(crate) fn stream_only(provider: &Provider, model: &str, upstream: &str) -> bool {
    let Some(value) = provider.preferences.get("upstream_stream_mode") else {
        return false;
    };
    value
        .get("models")
        .and_then(|m| m.get(model).or_else(|| m.get(upstream)))
        .or_else(|| value.get("default"))
        .unwrap_or(value)
        .as_str()
        == Some("stream_only")
}

pub(crate) fn responses_wire(engine: &str, url: &str) -> bool {
    engine.eq_ignore_ascii_case("codex")
        || (matches!(
            engine.to_ascii_lowercase().as_str(),
            "gpt" | "openrouter" | "azure" | "azure-databricks" | "cloudflare"
        ) && url.to_ascii_lowercase().contains("/responses"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn rejects_ambiguous_modes_and_scopes() {
        for value in [
            json!(true),
            json!(null),
            json!("auto"),
            json!({"default":false}),
            json!({"models":{"m":"force"}}),
            json!({"endpoints":{}}),
        ] {
            assert!(validate(&value).is_err(), "{value}");
        }
        for value in [
            json!("follow_client"),
            json!("stream_only"),
            json!({"default":"follow_client","models":{"m":"stream_only"}}),
        ] {
            assert!(validate(&value).is_ok());
        }
    }
}
