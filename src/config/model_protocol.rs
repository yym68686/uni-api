use crate::config::snapshot::Provider;
use serde_json::{json, Value};
use std::sync::Arc;
use url::Url;

pub(crate) const FIELD: &str = "model_upstream_protocols";
const INTERNAL: &str = "__model_upstream_protocols";
const PROTOCOLS: [&str; 4] = ["responses", "chat", "messages", "gemini"];

pub(crate) fn endpoint(protocol: &str) -> Option<(&'static str, &'static str)> {
    Some(match protocol {
        "responses" => ("codex", "/v1/responses"),
        "chat" => ("gpt", "/v1/chat/completions"),
        "messages" => ("claude", "/v1/messages"),
        "gemini" => ("gemini", "/v1beta"),
        _ => return None,
    })
}

/// Protocol switches can change only a known suffix, never the origin or
/// deployment prefix. Keep custom URLs usable through the inherited setting.
fn target(base: &str, engine: &str, protocol: &str) -> Option<(&'static str, String)> {
    let (next_engine, suffix) = endpoint(protocol)?;
    let mut url = Url::parse(base.trim()).ok()?;
    if !matches!(url.scheme(), "https" | "http")
        || !url.has_host()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let path = url.path().trim_end_matches('/');
    // A saved OpenRouter channel may have been imported with the System One
    // suffix. Its gateway inference must stay unchanged for other models, but
    // OpenRouter's documented OpenAI endpoints are safe same-origin choices.
    if url.scheme() == "https"
        && url.host_str() == Some("openrouter.ai")
        && url.port().is_none()
        && path == "/api/v1/systemone"
        && matches!(engine, "typesafe" | "openrouter" | "gpt")
    {
        if !matches!(protocol, "chat" | "responses") {
            return None;
        }
        url.set_path(&format!("/api{suffix}"));
        return Some((next_engine, url.into()));
    }
    if !matches!(engine, "codex" | "gpt" | "claude" | "gemini" | "openrouter") {
        return None;
    }
    let prefix = if path.is_empty() {
        ""
    } else {
        [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1beta/models",
            "/v1beta",
            "/v1",
        ]
        .into_iter()
        .find_map(|suffix| path.strip_suffix(suffix))?
    };
    if prefix.contains("/models/") || prefix.ends_with("/models") {
        return None;
    }
    url.set_path(&format!("{prefix}{suffix}"));
    Some((next_engine, url.into()))
}

pub(crate) fn options(provider: &Provider) -> Vec<&'static str> {
    if provider.api_keys.is_empty()
        || (provider.engine.as_ref() == "codex"
            && provider.api_keys.iter().any(|k| k.contains(',')))
    {
        return vec![];
    }
    PROTOCOLS
        .into_iter()
        .filter(|protocol| target(&provider.base_url, &provider.engine, protocol).is_some())
        .collect()
}

pub(crate) fn compile(
    raw: Option<&Value>,
    base: &str,
    engine: &str,
    api: Option<&Value>,
) -> Result<Value, String> {
    let mut compiled = serde_json::Map::new();
    let Some(raw) = raw else { return Ok(json!({})) };
    let entries = raw.as_object().ok_or("Invalid model_upstream_protocols")?;
    for (model, value) in entries {
        let protocol = value.as_str().ok_or("Invalid model upstream protocol")?;
        let (next_engine, url) = target(base, engine, protocol)
            .filter(|_| engine != "codex" || !api.is_some_and(composite_credentials))
            .ok_or("Model protocol requires a standard HTTP API-key provider and base path")?;
        if model.trim().is_empty() || model.len() > 256 || model.chars().any(char::is_control) {
            return Err("Invalid model upstream protocol model".into());
        }
        if protocol == "gemini" && !gemini_model(model) {
            return Err("Invalid Gemini upstream model path".into());
        }
        compiled.insert(model.clone(), json!([protocol, next_engine, url]));
    }
    Ok(Value::Object(compiled))
}

fn composite_credentials(api: &Value) -> bool {
    if let Some(keys) = api.as_array() {
        return keys.iter().any(composite_credentials);
    }
    api.as_str().is_some_and(|key| key.contains(','))
}

fn gemini_model(model: &str) -> bool {
    model.split('/').all(|segment| {
        !matches!(segment, "" | "." | "..")
            && segment
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
    })
}

pub(crate) fn attach(preferences: &mut serde_json::Map<String, Value>, compiled: Value) {
    preferences.remove(INTERNAL);
    if compiled.as_object().is_some_and(|m| !m.is_empty()) {
        preferences.insert(INTERNAL.to_owned(), compiled);
    }
}

fn selection<'a>(provider: &'a Provider, model: &str) -> Option<&'a [Value]> {
    let upstream = provider.models.get(model)?;
    provider
        .preferences
        .get(INTERNAL)?
        .get(upstream)?
        .as_array()
        .map(Vec::as_slice)
}

pub(crate) fn selected<'a>(provider: &'a Provider, model: &str) -> Option<&'a str> {
    selection(provider, model)?.first()?.as_str()
}

pub(crate) fn export(provider: &Provider) -> Option<Value> {
    provider.preferences.get(INTERNAL)?;
    let mut models = serde_json::Map::new();
    for (model, upstream) in provider.models.iter() {
        if let Some([_, _, url]) = selection(provider, model) {
            models.insert(
                model.clone(),
                json!({"base_url":url,"upstream_model":upstream}),
            );
        }
    }
    Some(Value::Object(models))
}

pub(crate) fn effective(mut provider: Arc<Provider>, model: &str) -> Arc<Provider> {
    if let Some([_, engine, url]) = selection(&provider, model) {
        let engine: Arc<str> = engine.as_str().unwrap_or_default().into();
        let url: Arc<str> = url.as_str().unwrap_or_default().into();
        let current = Arc::make_mut(&mut provider);
        current.engine = engine;
        current.base_url = url;
        current.engine_mode = "explicit".into();
    }
    provider
}

pub(crate) fn diagnostic(
    mut provider: Arc<Provider>,
    model: &str,
    protocol: &str,
) -> Option<Arc<Provider>> {
    if provider.engine.as_ref() == "codex" && provider.api_keys.iter().any(|k| k.contains(',')) {
        return None;
    }
    if protocol == "gemini" && !gemini_model(provider.models.get(model)?) {
        return None;
    }
    let (engine, url) = target(&provider.base_url, &provider.engine, protocol)?;
    let current = Arc::make_mut(&mut provider);
    current.engine = engine.into();
    current.base_url = url.into();
    Some(provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{compiler::compile_provider, snapshot::runtime_provider};

    #[test]
    fn keeps_identity_aliases_prefix_and_unrelated_models() {
        let document = json!({
            "provider": "manual",
            "engine": "codex",
            "base_url": "https://example.test/tenant%2Fwest/api/v1/responses",
            "api": "fixture",
            "model": [{"upstream-a":"alias-a"}, {"upstream-a":"alias-b"}, "unchanged"],
            "model_upstream_protocols": {"upstream-a": "messages"}
        });
        let provider = runtime_provider(
            serde_json::from_value(compile_provider(&document).unwrap()).unwrap(),
            Arc::default(),
        );
        for name in ["alias-a", "alias-b"] {
            let route = effective(provider.clone(), name);
            assert_eq!(route.name, provider.name);
            assert!(Arc::ptr_eq(&route.cursor, &provider.cursor));
            assert_eq!(route.models.get(name).unwrap(), "upstream-a");
            assert_eq!(route.engine.as_ref(), "claude");
            assert_eq!(
                route.base_url.as_ref(),
                "https://example.test/tenant%2Fwest/api/v1/messages"
            );
        }
        assert!(Arc::ptr_eq(
            &provider,
            &effective(provider.clone(), "unchanged")
        ));
        assert_eq!(provider.engine.as_ref(), "codex");
    }

    #[test]
    fn refuses_nonstandard_paths_and_invalid_credentials_before_switching() {
        for base in [
            "http://user@example.test/v1",
            "https://example.test/custom?key=fixture",
            "https://example.test/v1#fragment",
            "https://example.test/custom",
        ] {
            assert!(target(base, "gpt", "gemini").is_none(), "{base}");
        }
        let opts = json!({"a": "gemini"});
        assert!(compile(Some(&opts), "https://example.test/v1", "aws", None).is_err());
        assert!(compile(
            Some(&opts),
            "https://example.test/v1",
            "codex",
            Some(&json!(["refresh,account"]))
        )
        .is_err());
        assert!(compile(
            Some(&json!({"../b": "gemini"})),
            "https://example.test/v1",
            "gpt",
            None
        )
        .is_err());
        assert!(compile(
            Some(&json!({"a": "unsupported"})),
            "https://example.test/v1",
            "gpt",
            None
        )
        .is_err());
    }

    #[test]
    fn openrouter_legacy_systemone_can_select_its_documented_endpoints() {
        let document = json!({
            "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1/systemone",
            "api": "fixture", "model": ["jev-latest", "gemini-4-carbon"],
            "model_upstream_protocols": {"gemini-4-carbon": "chat"}
        });
        let provider = runtime_provider(
            serde_json::from_value(compile_provider(&document).unwrap()).unwrap(),
            Arc::default(),
        );
        assert_eq!(options(&provider), vec!["responses", "chat"]);
        assert_eq!(
            effective(provider.clone(), "gemini-4-carbon")
                .base_url
                .as_ref(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            effective(provider.clone(), "jev-latest").engine.as_ref(),
            "typesafe"
        );
        assert_eq!(
            diagnostic(provider.clone(), "gemini-4-carbon", "responses")
                .unwrap()
                .base_url
                .as_ref(),
            "https://openrouter.ai/api/v1/responses"
        );
        assert!(diagnostic(provider, "gemini-4-carbon", "gemini").is_none());
        assert!(target("https://api.typesafe.ai/v1/systemone", "typesafe", "chat").is_none());
        assert!(target(
            "https://openrouter.ai.example/api/v1/systemone",
            "typesafe",
            "chat"
        )
        .is_none());
    }
}
