use crate::config::compiler::compile_snapshot_bytes;
use crate::config::compiler::expand_environment;
use crate::config::compiler::infer_engine;
use crate::config::compiler::scalar_string;
use futures_util::future::join_all;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use url::Url;

pub(crate) async fn compile_snapshot_with_discovery(
    raw: &[u8],
    database_disabled: bool,
    client: &reqwest::Client,
    discovery_cache: &tokio::sync::Mutex<HashMap<String, Vec<String>>>,
) -> Result<Vec<u8>, String> {
    let yaml: serde_yaml::Value = serde_yaml::from_slice(raw)
        .map_err(|error| format!("decode uni-api YAML configuration: {error}"))?;
    let mut config = serde_json::to_value(yaml)
        .map_err(|error| format!("convert uni-api configuration: {error}"))?;
    expand_environment(&mut config);
    discover_missing_provider_models(&mut config, client, discovery_cache).await;
    let hydrated = serde_yaml::to_string(&config)
        .map_err(|error| format!("encode discovered runtime configuration: {error}"))?;
    compile_snapshot_bytes(hydrated.as_bytes(), database_disabled)
}

pub(crate) async fn discover_missing_provider_models(
    config: &mut Value,
    client: &reqwest::Client,
    discovery_cache: &tokio::sync::Mutex<HashMap<String, Vec<String>>>,
) {
    let Some(providers) = config.get_mut("providers").and_then(Value::as_array_mut) else {
        return;
    };
    let cached = discovery_cache.lock().await.clone();
    let mut pending = Vec::new();
    for (index, provider) in providers.iter_mut().enumerate() {
        if provider
            .get("model")
            .and_then(Value::as_array)
            .is_some_and(|models| !models.is_empty())
        {
            continue;
        }
        let cache_key = discovery_cache_key(provider);
        if let Some(models) = cached.get(&cache_key).filter(|models| !models.is_empty()) {
            provider
                .as_object_mut()
                .expect("provider configuration object")
                .insert(
                    "model".into(),
                    Value::Array(models.iter().cloned().map(Value::String).collect()),
                );
            continue;
        }
        pending.push({
            let provider = provider.clone();
            let client = client.clone();
            async move {
                (
                    index,
                    cache_key,
                    discover_provider_models(&client, &provider).await,
                )
            }
        });
    }
    let mut discovered = Vec::new();
    for (index, cache_key, result) in join_all(pending).await {
        match result {
            Ok(models) if !models.is_empty() => {
                discovered.push((cache_key, models.clone()));
                providers[index]
                    .as_object_mut()
                    .expect("provider configuration object")
                    .insert(
                        "model".into(),
                        Value::Array(models.into_iter().map(Value::String).collect()),
                    );
            }
            Ok(_) => {}
            Err(error) => eprintln!(
                "{}",
                json!({
                    "event_type": "rust_provider_model_discovery_error",
                    "provider": providers[index].get("provider").and_then(Value::as_str),
                    "error": error,
                })
            ),
        }
    }
    if !discovered.is_empty() {
        discovery_cache.lock().await.extend(discovered);
    }
}

pub(crate) fn discovery_cache_key(provider: &Value) -> String {
    format!("{:x}", Sha256::digest(provider.to_string().as_bytes()))
}

pub(crate) async fn discover_provider_models(
    client: &reqwest::Client,
    provider: &Value,
) -> Result<Vec<String>, String> {
    let base_url = provider
        .get("base_url")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if base_url.is_empty() {
        return Err("provider base_url is required for model discovery".into());
    }
    if base_url.contains("models.inference.ai.azure.com") {
        return Ok([
            "gpt-4o",
            "gpt-4.1",
            "gpt-4o-mini",
            "o4-mini",
            "o3",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect());
    }
    let api_key = first_provider_key(provider.get("api"));
    let engine = provider
        .get("engine")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| infer_engine(base_url));
    let mut headers = HeaderMap::new();
    let url = if engine == "gemini" {
        let before = base_url.split("/v1beta").next().unwrap_or(base_url);
        let mut url = Url::parse(&format!("{}/v1beta/models", before.trim_end_matches('/')))
            .map_err(|error| format!("invalid Gemini model discovery URL: {error}"))?;
        if !api_key.is_empty() {
            url.query_pairs_mut().append_pair("key", &api_key);
        }
        url
    } else {
        if !api_key.is_empty() {
            if engine == "claude" || engine == "vertex-claude" {
                headers.insert(
                    "x-api-key",
                    HeaderValue::from_str(&api_key)
                        .map_err(|_| "provider API key is not a valid header".to_owned())?,
                );
                headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            } else if engine == "azure" {
                headers.insert(
                    "api-key",
                    HeaderValue::from_str(&api_key)
                        .map_err(|_| "provider API key is not a valid header".to_owned())?,
                );
            } else {
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer {api_key}"))
                        .map_err(|_| "provider API key is not a valid header".to_owned())?,
                );
            }
        }
        provider_models_url(base_url)?
    };
    let response = client
        .get(url.clone())
        .headers(headers)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .map_err(|error| format!("fetch {url}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "model discovery returned HTTP {} from {url}",
            response.status().as_u16()
        ));
    }
    let payload = response
        .json::<Value>()
        .await
        .map_err(|error| format!("decode model discovery response from {url}: {error}"))?;
    Ok(discovered_model_ids(&payload))
}

pub(crate) fn first_provider_key(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.trim().to_owned(),
        Some(Value::Array(values)) => values
            .iter()
            .find_map(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        Some(value) => scalar_string(value).trim().to_owned(),
        None => String::new(),
    }
}

pub(crate) fn provider_models_url(base_url: &str) -> Result<Url, String> {
    let mut url = Url::parse(base_url)
        .map_err(|error| format!("invalid provider model discovery URL: {error}"))?;
    let known = [
        "/systemone",
        "/chat/completions",
        "/responses/compact",
        "/responses",
        "/messages",
        "/embeddings",
        "/moderations",
    ];
    let mut path = url.path().trim_end_matches('/').to_owned();
    for suffix in known {
        if path.ends_with(suffix) {
            path.truncate(path.len() - suffix.len());
            break;
        }
    }
    if path.ends_with("/v1") {
        path.push_str("/models");
    } else {
        path.push_str("/v1/models");
    }
    url.set_path(&path);
    Ok(url)
}

pub(crate) fn discovered_model_ids(payload: &Value) -> Vec<String> {
    let items = payload
        .get("data")
        .or_else(|| payload.get("models"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    let mut seen = std::collections::BTreeSet::new();
    for item in items {
        let value = item
            .get("id")
            .or_else(|| item.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim_start_matches("models/")
            .trim();
        if !value.is_empty() {
            seen.insert(value.to_owned());
        }
    }
    seen.into_iter().collect()
}
