use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::join_all;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use url::Url;

const SNAPSHOT_SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug)]
pub enum RuntimeConfigSource {
    File(PathBuf),
    Url(String),
}

impl RuntimeConfigSource {
    pub fn discover() -> Result<Self, String> {
        let configured_path =
            std::env::var("UNI_API_CONFIG_PATH").unwrap_or_else(|_| "api.yaml".to_owned());
        let path = PathBuf::from(configured_path);
        if path.is_file() {
            return Ok(Self::File(path));
        }
        let configured = std::env::var("CONFIG_URL").unwrap_or_default();
        let trimmed = configured.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "uni-api configuration is unavailable: {} does not exist and CONFIG_URL is unset",
                path.display()
            ));
        }
        if let Some(local) = trimmed.strip_prefix("file://") {
            return Ok(Self::File(PathBuf::from(local)));
        }
        if Path::new(trimmed).is_file() {
            return Ok(Self::File(PathBuf::from(trimmed)));
        }
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            return Ok(Self::Url(trimmed.to_owned()));
        }
        Err("CONFIG_URL must be an existing path, file:// URL, or HTTP(S) URL".into())
    }

    pub fn poll_interval(&self) -> std::time::Duration {
        match self {
            Self::File(_) => std::time::Duration::from_secs(2),
            Self::Url(_) => std::time::Duration::from_secs(30),
        }
    }

    pub async fn read(&self, client: &reqwest::Client) -> Result<Vec<u8>, String> {
        match self {
            Self::File(path) => tokio::fs::read(path)
                .await
                .map_err(|error| format!("read {}: {error}", path.display())),
            Self::Url(url) => {
                let response = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|error| format!("fetch runtime configuration: {error}"))?;
                if !response.status().is_success() {
                    return Err(format!(
                        "fetch runtime configuration returned HTTP {}",
                        response.status().as_u16()
                    ));
                }
                response
                    .bytes()
                    .await
                    .map(|bytes| bytes.to_vec())
                    .map_err(|error| format!("read runtime configuration response: {error}"))
            }
        }
    }
}

#[derive(Clone)]
pub struct RuntimeConfigPublisher {
    source: RuntimeConfigSource,
    client: reqwest::Client,
    snapshot_path: Arc<PathBuf>,
    database_disabled: bool,
    discovery_cache: Arc<tokio::sync::Mutex<HashMap<String, Vec<String>>>>,
}

impl RuntimeConfigPublisher {
    pub fn discover(database_disabled: bool) -> Result<Self, String> {
        let source = RuntimeConfigSource::discover()?;
        let client = reqwest::Client::builder()
            .http1_only()
            .build()
            .map_err(|error| format!("build configuration client: {error}"))?;
        let snapshot_path = std::env::var("RUST_RESPONSES_CONFIG_SNAPSHOT_PATH")
            .unwrap_or_else(|_| "/tmp/uni-api-rust-responses-config-v1.json".into());
        Ok(Self {
            source,
            client,
            snapshot_path: Arc::new(PathBuf::from(snapshot_path)),
            database_disabled,
            discovery_cache: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        })
    }

    pub async fn publish(&self) -> Result<(), String> {
        let raw = self.source.read(&self.client).await?;
        let snapshot = compile_snapshot_with_discovery(
            &raw,
            self.database_disabled,
            &self.client,
            &self.discovery_cache,
        )
        .await?;
        atomic_write(self.snapshot_path.as_ref(), &snapshot).await
    }

    pub async fn apply_patch(&self, patch: &Value) -> Result<(), String> {
        let RuntimeConfigSource::File(path) = &self.source else {
            return Err("runtime configuration is read-only when CONFIG_URL is remote".into());
        };
        let raw = tokio::fs::read(path)
            .await
            .map_err(|error| format!("read {} for update: {error}", path.display()))?;
        let yaml: serde_yaml::Value = serde_yaml::from_slice(&raw)
            .map_err(|error| format!("decode configuration for update: {error}"))?;
        let mut config = serde_json::to_value(yaml)
            .map_err(|error| format!("convert configuration for update: {error}"))?;
        let root = config
            .as_object_mut()
            .ok_or_else(|| "uni-api configuration must be a mapping".to_owned())?;
        let patch = patch
            .as_object()
            .ok_or_else(|| "configuration patch must be an object".to_owned())?;
        for (key, value) in patch {
            if matches!(
                key.as_str(),
                "providers" | "api_keys" | "preferences" | "video"
            ) {
                root.insert(key.clone(), value.clone());
            }
        }
        compile_snapshot_bytes(
            serde_yaml::to_string(&config)
                .map_err(|error| format!("validate configuration update: {error}"))?
                .as_bytes(),
            self.database_disabled,
        )?;
        write_config_file(path, &config).await?;
        self.publish().await
    }

    pub async fn add_credits(&self, paid_key: &str, amount: f64) -> Result<f64, String> {
        if !amount.is_finite() || amount <= 0.0 {
            return Err("amount must be positive".into());
        }
        let RuntimeConfigSource::File(path) = &self.source else {
            return Err("runtime configuration is read-only when CONFIG_URL is remote".into());
        };
        let raw = tokio::fs::read(path)
            .await
            .map_err(|error| format!("read {} for credits update: {error}", path.display()))?;
        let yaml: serde_yaml::Value = serde_yaml::from_slice(&raw)
            .map_err(|error| format!("decode configuration for credits update: {error}"))?;
        let mut config = serde_json::to_value(yaml)
            .map_err(|error| format!("convert configuration for credits update: {error}"))?;
        let keys = config
            .get_mut("api_keys")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "configuration contains no API keys".to_owned())?;
        let item = keys
            .iter_mut()
            .find(|item| item.get("api").and_then(Value::as_str) == Some(paid_key))
            .ok_or_else(|| "Paid API key not found".to_owned())?;
        let preferences = item
            .as_object_mut()
            .expect("API key configuration object")
            .entry("preferences")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| "API key preferences must be an object".to_owned())?;
        let current = preferences
            .get("credits")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let updated = current + amount;
        preferences.insert("credits".into(), json!(updated));
        write_config_file(path, &config).await?;
        self.publish().await?;
        Ok(updated)
    }

    pub fn start_watcher(&self) {
        let publisher = self.clone();
        tokio::spawn(async move {
            let mut last_revision = String::new();
            loop {
                let snapshot = match publisher.source.read(&publisher.client).await {
                    Ok(raw) => {
                        compile_snapshot_with_discovery(
                            &raw,
                            publisher.database_disabled,
                            &publisher.client,
                            &publisher.discovery_cache,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                };
                match snapshot {
                    Ok(snapshot) => {
                        let revision = serde_json::from_slice::<Value>(&snapshot)
                            .ok()
                            .and_then(|value| value.get("revision")?.as_str().map(str::to_owned))
                            .unwrap_or_default();
                        if revision != last_revision {
                            match atomic_write(publisher.snapshot_path.as_ref(), &snapshot).await {
                                Ok(()) => last_revision = revision,
                                Err(error) => eprintln!(
                                    "{}",
                                    json!({
                                        "event_type": "rust_runtime_config_publish_error",
                                        "error": error,
                                    })
                                ),
                            }
                        }
                    }
                    Err(error) => eprintln!(
                        "{}",
                        json!({
                            "event_type": "rust_runtime_config_reload_error",
                            "error": error,
                        })
                    ),
                }
                tokio::time::sleep(publisher.source.poll_interval()).await;
            }
        });
    }
}

async fn compile_snapshot_with_discovery(
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

async fn discover_missing_provider_models(
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

fn discovery_cache_key(provider: &Value) -> String {
    format!("{:x}", Sha256::digest(provider.to_string().as_bytes()))
}

async fn discover_provider_models(
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

fn first_provider_key(value: Option<&Value>) -> String {
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

fn provider_models_url(base_url: &str) -> Result<Url, String> {
    let mut url = Url::parse(base_url)
        .map_err(|error| format!("invalid provider model discovery URL: {error}"))?;
    let known = [
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

fn discovered_model_ids(payload: &Value) -> Vec<String> {
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

async fn write_config_file(path: &Path, config: &Value) -> Result<(), String> {
    let encoded = serde_yaml::to_string(config)
        .map_err(|error| format!("encode updated configuration: {error}"))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let temporary = parent.join(format!(
        ".{}.{}.update",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("api.yaml"),
        std::process::id(),
    ));
    tokio::fs::write(&temporary, encoded.as_bytes())
        .await
        .map_err(|error| format!("write temporary configuration: {error}"))?;
    match tokio::fs::rename(&temporary, path).await {
        Ok(()) => Ok(()),
        Err(_) => {
            let _ = tokio::fs::remove_file(&temporary).await;
            tokio::fs::write(path, encoded.as_bytes())
                .await
                .map_err(|error| format!("write updated configuration {}: {error}", path.display()))
        }
    }
}

async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| format!("create snapshot directory {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("snapshot"),
        std::process::id(),
    ));
    tokio::fs::write(&temporary, bytes)
        .await
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600))
            .await
            .map_err(|error| format!("chmod {}: {error}", temporary.display()))?;
    }
    tokio::fs::rename(&temporary, path)
        .await
        .map_err(|error| format!("publish {}: {error}", path.display()))
}

pub fn compile_snapshot_bytes(raw: &[u8], database_disabled: bool) -> Result<Vec<u8>, String> {
    let yaml: serde_yaml::Value = serde_yaml::from_slice(raw)
        .map_err(|error| format!("decode uni-api YAML configuration: {error}"))?;
    let mut config = serde_json::to_value(yaml)
        .map_err(|error| format!("convert uni-api configuration: {error}"))?;
    expand_environment(&mut config);
    let root = config
        .as_object()
        .ok_or_else(|| "uni-api configuration must be a mapping".to_owned())?;

    let providers = root
        .get("providers")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(compile_provider)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut api_keys = root
        .get("api_keys")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| compile_api_key(item, database_disabled))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    expand_api_key_aliases(&mut api_keys)?;
    if api_keys.is_empty() {
        return Err("uni-api configuration contains no usable API keys".into());
    }
    if providers.is_empty() {
        return Err("uni-api configuration contains no usable providers".into());
    }

    let mut snapshot = json!({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_unix_ms": unix_millis(),
        "database_disabled": database_disabled,
        "preferences": root.get("preferences").cloned().unwrap_or_else(|| json!({})),
        "api_keys": api_keys,
        "providers": providers,
        "api_config": config,
    });
    let mut revision_payload = snapshot.clone();
    revision_payload
        .as_object_mut()
        .expect("snapshot object")
        .remove("generated_unix_ms");
    let canonical = serde_json::to_vec(&revision_payload)
        .map_err(|error| format!("encode runtime configuration revision: {error}"))?;
    let revision = format!("{:x}", Sha256::digest(canonical));
    snapshot
        .as_object_mut()
        .expect("snapshot object")
        .insert("revision".into(), Value::String(revision));
    serde_json::to_vec(&snapshot)
        .map_err(|error| format!("encode compiled runtime configuration: {error}"))
}

fn compile_provider(value: &Value) -> Option<Value> {
    let item = value.as_object()?;
    let name = scalar_string(item.get("provider")?).trim().to_owned();
    if name.is_empty() {
        return None;
    }
    // Keep the configured URL byte-for-byte.  The Python runtime also keeps
    // proxy/gateway URLs here and derives the provider protocol separately.
    // Replacing a project-backed URL with the public Google endpoint would
    // silently bypass an operator-configured gateway.
    let base_url = scalar_string(item.get("base_url").unwrap_or(&Value::Null));
    let (models, model_order) = compile_models(item.get("model"));
    if models.is_empty() {
        return None;
    }
    let mut preferences = item
        .get("preferences")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !preferences.contains_key("max_request_body_bytes") {
        if let Some(limit) = item.get("max_request_body_bytes") {
            preferences.insert("max_request_body_bytes".into(), limit.clone());
        }
    }
    let engine = item
        .get("engine")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| infer_engine(&base_url));
    let aws_access_key = scalar_string(item.get("aws_access_key").unwrap_or(&Value::Null));
    let aws_secret_key = scalar_string(item.get("aws_secret_key").unwrap_or(&Value::Null));
    let aws_session_token = scalar_string(item.get("aws_session_token").unwrap_or(&Value::Null));
    let client_email = scalar_string(item.get("client_email").unwrap_or(&Value::Null));
    let private_key = scalar_string(item.get("private_key").unwrap_or(&Value::Null));
    let provider_api = match item.get("api") {
        Some(value) if !value.is_null() => value.clone(),
        _ if engine.eq_ignore_ascii_case("aws") && !aws_access_key.trim().is_empty() => {
            Value::String(aws_access_key.clone())
        }
        _ if matches!(
            engine.to_ascii_lowercase().as_str(),
            "vertex" | "vertex-gemini" | "vertex-claude"
        ) && !client_email.trim().is_empty()
            && !private_key.trim().is_empty() =>
        {
            Value::String("__vertex_oauth__".into())
        }
        _ => Value::Null,
    };
    let region = scalar_string(item.get("region").unwrap_or(&Value::Null));
    let region = if region.is_empty() {
        "global".to_owned()
    } else {
        region
    };
    Some(json!({
        "name": name,
        "base_url": base_url,
        "engine": engine,
        "api": provider_api,
        "project_id": scalar_string(item.get("project_id").unwrap_or(&Value::Null)),
        "private_key": private_key,
        "client_email": client_email,
        "aws_access_key": aws_access_key,
        "aws_secret_key": aws_secret_key,
        "aws_session_token": aws_session_token,
        "cf_account_id": scalar_string(item.get("cf_account_id").unwrap_or(&Value::Null)),
        "region": region,
        "models": models,
        "model_order": model_order,
        "preferences": preferences,
        "exclude_endpoints": merge_endpoint_values(
            item.get("exclude_endpoints"),
            preferences.get("exclude_endpoints"),
        ),
        "only_request_types": item.get("only_request_types").cloned().unwrap_or(Value::Null),
        "exclude_request_types": item.get("exclude_request_types").cloned().unwrap_or(Value::Null),
        "exclude_request_rules": merge_rule_values(
            item.get("exclude_request_rules"),
            preferences.get("exclude_request_rules"),
        ),
    }))
}

fn infer_engine(base_url: &str) -> String {
    let lower = base_url.trim().to_ascii_lowercase();
    if lower.contains("/v1/messages") || lower.contains("/claude/") {
        return "claude".into();
    }
    if lower.contains("/v1beta") || lower.contains("generativelanguage.googleapis.com") {
        return "gemini".into();
    }
    if lower.contains("aiplatform.googleapis.com") {
        return "vertex".into();
    }
    if lower.contains("amazonaws.com") {
        return "aws".into();
    }
    if lower.contains("api.cohere.com") {
        return "cohere".into();
    }
    "gpt".into()
}

fn expand_api_key_aliases(api_keys: &mut [Value]) -> Result<(), String> {
    let mut by_token = BTreeMap::new();
    for (index, item) in api_keys.iter().enumerate() {
        if let Some(token) = item
            .get("token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            by_token.insert(token.to_owned(), index);
        }
    }
    let original = api_keys.to_vec();
    for index in 0..api_keys.len() {
        let rules = original[index]
            .get("model_rules")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut expanded = Vec::new();
        let mut visiting = Vec::new();
        for rule in rules {
            expand_rule(&rule, &original, &by_token, &mut visiting, &mut expanded)?;
        }
        if expanded.is_empty() {
            expanded.push(Value::String("all".into()));
        }
        api_keys[index]
            .as_object_mut()
            .expect("compiled API key is an object")
            .insert(
                "model_rules".into(),
                Value::Array(deduplicate_values(expanded)),
            );
    }
    Ok(())
}

fn expand_rule(
    rule: &Value,
    api_keys: &[Value],
    by_token: &BTreeMap<String, usize>,
    visiting: &mut Vec<String>,
    output: &mut Vec<Value>,
) -> Result<(), String> {
    let Some(raw) = rule
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let Some((alias, requested_model)) = raw.split_once('/') else {
        output.push(Value::String(raw.to_owned()));
        return Ok(());
    };
    let Some(alias_index) = by_token.get(alias).copied() else {
        output.push(Value::String(raw.to_owned()));
        return Ok(());
    };
    if visiting.iter().any(|item| item == alias) {
        return Err(format!("cyclic API key model alias involving {alias}"));
    }
    visiting.push(alias.to_owned());
    let alias_rules = api_keys[alias_index]
        .get("model_rules")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for alias_rule in alias_rules {
        let Some(alias_rule_text) = alias_rule.as_str() else {
            continue;
        };
        if requested_model == "*" {
            expand_rule(
                &Value::String(alias_rule_text.to_owned()),
                api_keys,
                by_token,
                visiting,
                output,
            )?;
            continue;
        }
        let matches = alias_rule_text == requested_model
            || alias_rule_text.ends_with("/*")
            || alias_rule_text.ends_with(&format!("/{requested_model}"));
        if matches {
            let concrete = if alias_rule_text.ends_with("/*") {
                alias_rule_text.trim_end_matches('*').to_owned() + requested_model
            } else {
                alias_rule_text.to_owned()
            };
            expand_rule(
                &Value::String(concrete),
                api_keys,
                by_token,
                visiting,
                output,
            )?;
        }
    }
    visiting.pop();
    Ok(())
}

fn deduplicate_values(values: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::BTreeSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(value.to_string()))
        .collect()
}

fn compile_api_key(value: &Value, _database_disabled: bool) -> Option<Value> {
    let item = value.as_object()?;
    let token = scalar_string(item.get("api")?).trim().to_owned();
    if token.is_empty() {
        return None;
    }
    let configured_rules = item
        .get("model")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![Value::String("all".into())]);
    let mut model_rules = Vec::with_capacity(configured_rules.len());
    let mut weights = item
        .get("weights")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for rule in configured_rules {
        if let Some(value) = rule.as_str() {
            model_rules.push(Value::String(value.to_owned()));
            continue;
        }
        let Some(object) = rule.as_object() else {
            continue;
        };
        if let Some((name, weight)) = object.iter().next() {
            model_rules.push(Value::String(name.clone()));
            weights.insert(name.clone(), weight.clone());
        }
    }
    if model_rules.is_empty() {
        model_rules.push(Value::String("all".into()));
    }
    let preferences = item
        .get("preferences")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    Some(json!({
        "token": token,
        "model_rules": model_rules,
        "role": item.get("role").and_then(Value::as_str).unwrap_or_else(|| token.get(..8).unwrap_or(&token)),
        "weights": weights,
        "preferences": preferences,
        "native_paid_state_safe": true,
    }))
}

fn compile_models(value: Option<&Value>) -> (BTreeMap<String, String>, Vec<String>) {
    let mut models = BTreeMap::new();
    let mut order = Vec::new();
    for item in value.and_then(Value::as_array).into_iter().flatten() {
        if let Some(model) = item.as_str() {
            if !models.contains_key(model) {
                order.push(model.to_owned());
            }
            models.insert(model.to_owned(), model.to_owned());
            continue;
        }
        let Some(mapping) = item.as_object() else {
            continue;
        };
        for (upstream, exposed) in mapping {
            let exposed = scalar_string(exposed);
            if exposed.is_empty() {
                continue;
            }
            if !models.contains_key(&exposed) {
                order.push(exposed.clone());
            }
            models.insert(exposed, upstream.clone());
        }
    }
    (models, order)
}

fn merge_endpoint_values(first: Option<&Value>, second: Option<&Value>) -> Vec<Value> {
    let mut values = Vec::new();
    for source in [first, second].into_iter().flatten() {
        if let Some(items) = source.as_array() {
            values.extend(items.iter().cloned());
        } else if !source.is_null() {
            values.push(source.clone());
        }
    }
    values
}

fn merge_rule_values(first: Option<&Value>, second: Option<&Value>) -> Vec<Value> {
    let mut values = Vec::new();
    for source in [first, second].into_iter().flatten() {
        if let Some(items) = source.as_array() {
            values.extend(items.iter().filter(|item| item.is_object()).cloned());
        } else if source.is_object() {
            values.push(source.clone());
        }
    }
    values
}

fn scalar_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        _ => String::new(),
    }
}

fn expand_environment(value: &mut Value) {
    match value {
        Value::Object(object) => object.values_mut().for_each(expand_environment),
        Value::Array(items) => items.iter_mut().for_each(expand_environment),
        Value::String(text) => *text = expand_string(text),
        _ => {}
    }
}

fn expand_string(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = String::with_capacity(value.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        let Some(relative) = value[cursor..].find("${") else {
            output.push_str(&value[cursor..]);
            break;
        };
        let start = cursor + relative;
        output.push_str(&value[cursor..start]);
        let Some(end_relative) = value[start + 2..].find('}') else {
            output.push_str(&value[start..]);
            break;
        };
        let end = start + 2 + end_relative;
        let expression = &value[start + 2..end];
        let (name, default) = expression
            .split_once(":-")
            .map(|(name, default)| (name, Some(default)))
            .unwrap_or((expression, None));
        let replacement = std::env::var(name)
            .ok()
            .or_else(|| default.map(str::to_owned));
        if let Some(replacement) = replacement {
            output.push_str(&replacement);
        }
        cursor = end + 1;
    }
    output
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_python_compatible_model_mappings_and_paid_state() {
        let raw = br#"
providers:
  - provider: codex-a
    base_url: https://example.com/v1/responses
    engine: codex
    api: secret-upstream
    model:
      - upstream-sol: gpt-5.6-sol
api_keys:
  - api: client-key
    model:
      - codex-a/*
    preferences:
      rate_limit: 10/min
"#;
        let bytes = compile_snapshot_bytes(raw, false).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["providers"][0]["models"]["gpt-5.6-sol"],
            "upstream-sol"
        );
        assert_eq!(value["providers"][0]["model_order"][0], "gpt-5.6-sol");
        assert_eq!(value["api_keys"][0]["native_paid_state_safe"], true);
        assert_eq!(value["revision"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn preserves_provider_exclude_request_rules() {
        let raw = br#"
providers:
  - provider: codex-a
    base_url: https://example.com/v1/responses
    engine: codex
    api: secret-upstream
    model:
      - codex-auto-review: gpt-5.6-luna
    exclude_request_rules:
      - match:
          endpoint: /v1/responses
          request_model: gpt-5.6-luna
          upstream_model: codex-auto-review
          reasoning_effort: [max]
        reason: unsupported_reasoning_effort
api_keys:
  - api: client-key
    model: [codex-a/*]
"#;
        let value: Value =
            serde_json::from_slice(&compile_snapshot_bytes(raw, true).unwrap()).unwrap();
        assert_eq!(
            value["providers"][0]["exclude_request_rules"][0]["match"]["reasoning_effort"][0],
            "max"
        );
    }

    #[test]
    fn paid_key_is_admitted_to_native_database_balance_checks() {
        let raw = br#"
providers:
  - provider: p
    base_url: https://example.com/v1/responses
    api: upstream
    model: [m]
api_keys:
  - api: client
    model: [p/m]
    preferences:
      credits: 1.5
"#;
        let bytes = compile_snapshot_bytes(raw, false).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["api_keys"][0]["native_paid_state_safe"], true);
    }

    #[test]
    fn expands_nested_api_key_model_rules_and_infers_engines() {
        let raw = br#"
providers:
  - provider: claude-a
    base_url: https://example.com/v1/messages
    api: upstream
    model: [claude-opus]
api_keys:
  - api: child-key
    model: [claude-a/*]
  - api: parent-key
    model: [child-key/*]
"#;
        let bytes = compile_snapshot_bytes(raw, true).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["providers"][0]["engine"], "claude");
        assert_eq!(value["api_keys"][1]["model_rules"][0], "claude-a/*");
    }

    #[test]
    fn preserves_operator_gateway_for_project_backed_provider() {
        let raw = br#"
providers:
  - provider: vertex-through-gateway
    base_url: https://gateway.example/proxy/https://aiplatform.googleapis.com/
    project_id: project-a
    api: upstream
    model: [gemini]
api_keys:
  - api: client
    model: [vertex-through-gateway/*]
"#;
        let bytes = compile_snapshot_bytes(raw, true).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["providers"][0]["base_url"],
            "https://gateway.example/proxy/https://aiplatform.googleapis.com/"
        );
    }

    #[test]
    fn model_discovery_normalizes_openai_and_gemini_shapes() {
        assert_eq!(
            provider_models_url("https://example.com/v1/chat/completions")
                .unwrap()
                .as_str(),
            "https://example.com/v1/models"
        );
        assert_eq!(
            discovered_model_ids(&json!({
                "models":[
                    {"name":"models/gemini-2.5-pro"},
                    {"name":"models/gemini-2.5-flash"},
                    {"name":"models/gemini-2.5-pro"}
                ]
            })),
            vec!["gemini-2.5-flash", "gemini-2.5-pro"]
        );
    }

    #[test]
    fn credential_only_special_providers_receive_native_route_keys() {
        let raw = br#"
providers:
  - provider: bedrock
    base_url: https://bedrock-runtime.us-east-1.amazonaws.com
    engine: aws
    aws_access_key: AKIA_TEST
    aws_secret_key: secret
    model: [anthropic.claude]
  - provider: vertex
    base_url: https://aiplatform.googleapis.com
    engine: vertex-claude
    client_email: svc@example.com
    private_key: key
    project_id: project-a
    model: [claude]
api_keys:
  - api: client
    model: [bedrock/*, vertex/*]
"#;
        let value: Value =
            serde_json::from_slice(&compile_snapshot_bytes(raw, true).unwrap()).unwrap();
        assert_eq!(value["providers"][0]["api"], "AKIA_TEST");
        assert_eq!(value["providers"][1]["api"], "__vertex_oauth__");
        assert_eq!(value["providers"][0]["aws_secret_key"], "secret");
    }
}
