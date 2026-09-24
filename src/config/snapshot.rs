use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub(crate) const SNAPSHOT_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Deserialize)]
pub(crate) struct RawSnapshot {
    pub(crate) schema_version: u64,
    pub(crate) revision: String,
    #[serde(default)]
    pub(crate) preferences: Map<String, Value>,
    #[serde(default)]
    pub(crate) api_keys: Vec<RawApiKey>,
    #[serde(default)]
    pub(crate) providers: Vec<RawProvider>,
    #[serde(default)]
    pub(crate) api_config: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RawApiKey {
    pub(crate) token: String,
    #[serde(default)]
    pub(crate) model_rules: Vec<Value>,
    #[serde(default)]
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) weights: Map<String, Value>,
    #[serde(default)]
    pub(crate) preferences: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RawProvider {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) engine: Option<String>,
    pub(crate) api: Value,
    #[serde(default)]
    pub(crate) project_id: Option<String>,
    #[serde(default)]
    pub(crate) private_key: Option<String>,
    #[serde(default)]
    pub(crate) client_email: Option<String>,
    #[serde(default)]
    pub(crate) aws_access_key: Option<String>,
    #[serde(default)]
    pub(crate) aws_secret_key: Option<String>,
    #[serde(default)]
    pub(crate) aws_session_token: Option<String>,
    #[serde(default)]
    pub(crate) cf_account_id: Option<String>,
    #[serde(default)]
    pub(crate) region: Option<String>,
    #[serde(default)]
    pub(crate) models: HashMap<String, String>,
    #[serde(default)]
    pub(crate) preferences: Map<String, Value>,
    #[serde(default)]
    pub(crate) exclude_endpoints: Value,
    #[serde(default)]
    pub(crate) only_request_types: Value,
    #[serde(default)]
    pub(crate) exclude_request_types: Value,
    #[serde(default)]
    pub(crate) exclude_request_rules: Value,
}

#[derive(Clone)]
pub(crate) struct Snapshot {
    pub(crate) revision: Arc<str>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) api_keys: Arc<HashMap<String, Arc<ApiKey>>>,
    pub(crate) api_key_order: Arc<Vec<String>>,
    pub(crate) providers: Arc<Vec<Arc<Provider>>>,
    pub(crate) providers_by_name: Arc<HashMap<String, Arc<Provider>>>,
    pub(crate) api_config: Arc<Value>,
}

#[derive(Clone)]
pub(crate) struct ApiKey {
    pub(crate) token: Arc<str>,
    pub(crate) model_rules: Arc<Vec<String>>,
    pub(crate) role: Arc<str>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) weights: Arc<Map<String, Value>>,
    pub(crate) native_supported: bool,
}

#[derive(Clone)]
pub(crate) struct Provider {
    pub(crate) name: Arc<str>,
    pub(crate) base_url: Arc<str>,
    pub(crate) engine: Arc<str>,
    pub(crate) api_keys: Arc<Vec<String>>,
    pub(crate) project_id: Option<Arc<str>>,
    pub(crate) private_key: Option<Arc<str>>,
    pub(crate) client_email: Option<Arc<str>>,
    pub(crate) aws_access_key: Option<Arc<str>>,
    pub(crate) aws_secret_key: Option<Arc<str>>,
    pub(crate) aws_session_token: Option<Arc<str>>,
    pub(crate) cf_account_id: Option<Arc<str>>,
    pub(crate) region: Arc<str>,
    pub(crate) models: Arc<HashMap<String, String>>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) excluded_endpoints: Arc<Vec<String>>,
    pub(crate) only_request_types: Arc<Vec<String>>,
    pub(crate) excluded_request_types: Arc<Vec<String>>,
    pub(crate) excluded_request_rules: Arc<Vec<Value>>,
    pub(crate) cursor: Arc<AtomicUsize>,
}

pub(crate) fn runtime_provider(item: RawProvider, cursor: Arc<AtomicUsize>) -> Arc<Provider> {
    let name = item.name.trim().to_owned();
    Arc::new(Provider {
        name: name.clone().into(),
        base_url: item.base_url.trim().to_owned().into(),
        engine: item.engine.unwrap_or_else(|| "gpt".into()).into(),
        api_keys: Arc::new(provider_api_keys(&item.api)),
        project_id: item
            .project_id
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        private_key: item
            .private_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        client_email: item
            .client_email
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_access_key: item
            .aws_access_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_secret_key: item
            .aws_secret_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_session_token: item
            .aws_session_token
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        cf_account_id: item
            .cf_account_id
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        region: item
            .region
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "global".into())
            .into(),
        models: Arc::new(item.models),
        preferences: Arc::new(item.preferences),
        excluded_endpoints: Arc::new(endpoint_values(&item.exclude_endpoints)),
        only_request_types: Arc::new(request_type_values(&item.only_request_types)),
        excluded_request_types: Arc::new(request_type_values(&item.exclude_request_types)),
        excluded_request_rules: Arc::new(request_rule_values(&item.exclude_request_rules)),
        cursor,
    })
}

pub(crate) fn request_rule_values(value: &Value) -> Vec<Value> {
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .filter(|item| item.is_object())
            .cloned()
            .collect();
    }
    value
        .is_object()
        .then(|| value.clone())
        .into_iter()
        .collect()
}

pub(crate) fn provider_api_keys(value: &Value) -> Vec<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => vec![value.trim().to_owned()],
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn endpoint_values(value: &Value) -> Vec<String> {
    match value {
        Value::String(value) => vec![value.clone()],
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn request_type_values(value: &Value) -> Vec<String> {
    endpoint_values(value)
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}
