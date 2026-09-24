use crate::config::snapshot::ApiKey;
use crate::config::snapshot::Provider;
use crate::config::snapshot::Snapshot;
use crate::routing::access::extract_api_key;
use crate::routing::filters::provider_accepts_body;
use crate::routing::filters::provider_accepts_endpoint;
use crate::routing::filters::provider_accepts_request_rules;
use crate::routing::filters::provider_accepts_request_type;
use crate::routing::types::ResolvedRoute;
use crate::routing::types::RouteResolutionError;
use crate::runtime::scheduling::parse_rate_limits;
use crate::runtime::scheduling::tpr_exceeded;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::Persistence;
use crate::upstream::hedging::parse_hedging;
use crate::upstream::hedging::HedgingConfig;
use axum::http::{HeaderMap, StatusCode};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub(crate) static SCHEDULING_NONCE: AtomicU64 = AtomicU64::new(1);

impl GatewayRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn resolve_route(
        &self,
        persistence: &Persistence,
        headers: &HeaderMap,
        request_model: &str,
        endpoint: &str,
        request_body_bytes: u64,
        request_type: Option<&str>,
        admit_rate: bool,
    ) -> Result<ResolvedRoute, RouteResolutionError> {
        let snapshot = self.snapshot().await.ok_or_else(|| RouteResolutionError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Runtime configuration is not ready".into(),
        })?;
        let token = extract_api_key(headers).ok_or_else(|| RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Invalid or missing API Key".into(),
        })?;
        let api_key =
            snapshot
                .api_keys
                .get(&token)
                .cloned()
                .ok_or_else(|| RouteResolutionError {
                    status: StatusCode::FORBIDDEN,
                    message: "Invalid or missing API Key".into(),
                })?;
        if !api_key.native_supported {
            return Err(RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "API key contains an unsupported model rule".into(),
            });
        }
        self.ensure_paid_balance(persistence, &api_key).await?;
        let global_rules = parse_rate_limits(snapshot.preferences.get("rate_limit"), None)
            .ok_or_else(|| RouteResolutionError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "Invalid global rate-limit configuration".into(),
            })?;
        let client_rules =
            parse_rate_limits(api_key.preferences.get("rate_limit"), Some(request_model))
                .ok_or_else(|| RouteResolutionError {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    message: "Invalid client rate-limit configuration".into(),
                })?;
        let estimated_tokens = (request_body_bytes / 4).max(1) as usize;
        if tpr_exceeded(&client_rules, estimated_tokens) {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Tokens per request limit exceeded".into(),
            });
        }
        if admit_rate {
            for child in nested_keys_for_model(&snapshot, &api_key, request_model) {
                if let Some(rules) =
                    parse_rate_limits(child.preferences.get("rate_limit"), Some(request_model))
                {
                    if tpr_exceeded(&rules, estimated_tokens)
                        || !self
                            .admit_rate(&format!("client:{}", child.token), &rules)
                            .await
                    {
                        return Err(RouteResolutionError {
                            status: StatusCode::TOO_MANY_REQUESTS,
                            message: "Nested API-key rate limit exceeded".into(),
                        });
                    }
                }
            }
        }
        if admit_rate
            && (!self.admit_rate("__global__", &global_rules).await
                || !self
                    .admit_rate(&format!("client:{}", api_key.token), &client_rules)
                    .await)
        {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Too many requests".into(),
            });
        }
        let route_key = diagnostic_key(&snapshot, &api_key, headers, endpoint)?;
        let providers = matching_providers(
            &snapshot,
            &route_key,
            request_model,
            request_body_bytes,
            request_type,
            None,
            endpoint.trim_end_matches('/'),
        )
        .map_err(|_| RouteResolutionError {
            status: StatusCode::NOT_IMPLEMENTED,
            message: "Nested runtime provider is not available".into(),
        })?;
        if providers.is_empty() {
            return Err(RouteResolutionError {
                status: StatusCode::NOT_FOUND,
                message: format!("No available providers at the moment: {request_model}"),
            });
        }
        let providers = self
            .schedule_providers(&api_key, request_model, providers)
            .await;
        if providers.is_empty() {
            return Err(RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "All matching channels are temporarily disabled".into(),
            });
        }
        let hedging = if headers.contains_key(TARGET_PROVIDER_HEADER) {
            HedgingConfig::default()
        } else {
            parse_hedging(&snapshot.preferences)
        };
        Ok(ResolvedRoute { providers, hedging })
    }

    pub(crate) async fn schedule_providers(
        &self,
        api_key: &ApiKey,
        request_model: &str,
        providers: Vec<Arc<Provider>>,
    ) -> Vec<Arc<Provider>> {
        let controls = self.channel_controls.read().await.routing_rules();
        let key = crate::routing::catalog::key_id(&api_key.token);
        if controls.order(&key, request_model).is_some() {
            return controls.apply(&key, request_model, providers);
        }
        let scheduled = self
            .schedule_configured_providers(api_key, request_model, providers)
            .await;
        if controls.is_empty() {
            scheduled
        } else {
            controls.apply(&key, request_model, scheduled)
        }
    }

    async fn schedule_configured_providers(
        &self,
        api_key: &ApiKey,
        request_model: &str,
        providers: Vec<Arc<Provider>>,
    ) -> Vec<Arc<Provider>> {
        if providers.len() <= 1 {
            return providers;
        }
        let algorithm = api_key
            .preferences
            .get("SCHEDULING_ALGORITHM")
            .and_then(Value::as_str)
            .or_else(|| {
                api_key
                    .preferences
                    .get("api_key_schedule_algorithm")
                    .and_then(Value::as_str)
            })
            .unwrap_or("fixed_priority")
            .trim()
            .to_ascii_lowercase();
        let mut scheduled = weighted_provider_sequence(
            &providers,
            request_model,
            api_key.weights.as_ref(),
            &algorithm,
        );
        if scheduled.is_empty() {
            scheduled = providers;
        }
        if algorithm == "random" {
            shuffle_providers(
                &mut scheduled,
                scheduling_seed(&api_key.token, request_model),
            );
            return scheduled;
        }
        if algorithm == "fixed_priority" || scheduled.len() <= 1 {
            return scheduled;
        }
        let mut cursors = self.scheduling.routing_cursors.lock().await;
        let cursor = cursors
            .entry((api_key.token.to_string(), request_model.to_owned()))
            .or_insert(0);
        let start = *cursor % scheduled.len();
        *cursor = (start + 1) % scheduled.len();
        scheduled.rotate_left(start);
        scheduled
    }
}

pub(crate) fn nested_keys_for_model(
    snapshot: &Snapshot,
    key: &ApiKey,
    model: &str,
) -> Vec<Arc<ApiKey>> {
    fn walk(
        snapshot: &Snapshot,
        key: &ApiKey,
        model: &str,
        out: &mut Vec<Arc<ApiKey>>,
        seen: &mut std::collections::BTreeSet<String>,
    ) {
        if !seen.insert(key.token.to_string()) {
            return;
        }
        let rules = key
            .preferences
            .get("__route_graph")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        for rule in rules {
            let Some((alias, requested)) = rule.split_once('/') else {
                continue;
            };
            if (requested == "*" || requested == model) && snapshot.api_keys.get(alias).is_some() {
                let child = snapshot.api_keys.get(alias).unwrap();
                out.push(child.clone());
                walk(snapshot, child, model, out, seen);
            }
        }
        seen.remove(key.token.as_ref());
    }
    let mut out = Vec::new();
    walk(
        snapshot,
        key,
        model,
        &mut out,
        &mut std::collections::BTreeSet::new(),
    );
    out
}

// Explicit administrator-only diagnostic routing. The temporary key is never
// written to the snapshot; normal traffic retains its configured routing graph.
pub(crate) const TARGET_PROVIDER_HEADER: &str = "x-uni-api-provider";

pub(crate) const PROBE_UNCONFIGURED_MODEL_HEADER: &str = "x-uni-api-probe-unconfigured-model";

pub(crate) fn diagnostic_key(
    snapshot: &Snapshot,
    key: &ApiKey,
    headers: &HeaderMap,
    endpoint: &str,
) -> Result<ApiKey, RouteResolutionError> {
    let Some(value) = headers.get(TARGET_PROVIDER_HEADER) else {
        if headers.contains_key(PROBE_UNCONFIGURED_MODEL_HEADER) {
            return Err(RouteResolutionError {
                status: StatusCode::BAD_REQUEST,
                message: "Unconfigured model probes require a target provider".into(),
            });
        }
        return Ok(key.clone());
    };
    if !crate::routing::catalog::can_inspect_all(snapshot, key) {
        return Err(RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Targeted requests require a platform administrator key".into(),
        });
    }
    if endpoint.trim_end_matches('/') != "/v1/responses" {
        return Err(RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Targeted requests require /v1/responses".into(),
        });
    }
    let name = value
        .to_str()
        .ok()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Invalid target provider".into(),
        })?;
    if name.contains('/') || snapshot.api_keys.contains_key(name) {
        return Err(RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Ambiguous target provider name".into(),
        });
    }
    if !snapshot.providers_by_name.contains_key(name) {
        return Err(RouteResolutionError {
            status: StatusCode::NOT_FOUND,
            message: "Target provider not found".into(),
        });
    }
    let mut diagnostic = key.clone();
    diagnostic.model_rules = Arc::new(vec![format!("{name}/*")]);
    let mut preferences = (*key.preferences).clone();
    preferences.remove("__route_graph");
    preferences.insert("__diagnostic_provider".into(), json!(name));
    if let Some(value) = headers.get(PROBE_UNCONFIGURED_MODEL_HEADER) {
        if value != "true" {
            return Err(RouteResolutionError {
                status: StatusCode::BAD_REQUEST,
                message: "Invalid unconfigured model probe flag".into(),
            });
        }
        preferences.insert("__diagnostic_unconfigured_model".into(), json!(true));
    }
    diagnostic.preferences = Arc::new(preferences);
    Ok(diagnostic)
}

pub(crate) fn matching_providers(
    snapshot: &Snapshot,
    api_key: &ApiKey,
    request_model: &str,
    request_body_bytes: u64,
    request_type: Option<&str>,
    reasoning_effort: Option<&str>,
    endpoint: &str,
) -> Result<Vec<Arc<Provider>>, ()> {
    // Walk the key graph recursively. A child key contributes only the model
    // rules it explicitly owns; its token is never turned into a synthetic
    // upstream channel, so parent and child policy boundaries stay visible to
    // scheduling and accounting.
    fn collect(
        snapshot: &Snapshot,
        key: &ApiKey,
        model: &str,
        out: &mut Vec<Arc<Provider>>,
        visiting: &mut std::collections::BTreeSet<String>,
    ) {
        if !visiting.insert(key.token.to_string()) {
            return;
        }
        let rules = key
            .preferences
            .get("__route_graph")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_else(|| key.model_rules.iter().map(String::as_str).collect());
        for rule in rules {
            let Some((alias, requested)) = rule.split_once('/') else {
                if rule == "all"
                    || rule == model
                    || (rule.starts_with('<')
                        && rule.ends_with('>')
                        && &rule[1..rule.len() - 1] == model)
                {
                    out.extend(
                        snapshot
                            .providers
                            .iter()
                            .filter(|p| p.models.contains_key(model))
                            .cloned(),
                    );
                }
                continue;
            };
            if let Some(child) = snapshot.api_keys.get(alias) {
                if requested == "*" || requested == model {
                    collect(snapshot, child, model, out, visiting);
                }
                continue;
            }
            if let Some(provider) = snapshot.providers_by_name.get(alias) {
                if (requested == "*" || requested == model) && provider.models.contains_key(model) {
                    out.push(provider.clone());
                }
            }
        }
        visiting.remove(key.token.as_ref());
    }
    let mut matches = Vec::new();
    collect(
        snapshot,
        api_key,
        request_model,
        &mut matches,
        &mut std::collections::BTreeSet::new(),
    );
    // Only an authenticated diagnostic key can opt in. Extend a request-local
    // provider clone; preserve configured aliases and every other provider rule.
    // The live snapshot, caller model lists and future requests are untouched.
    if matches.is_empty()
        && api_key
            .preferences
            .get("__diagnostic_unconfigured_model")
            .and_then(Value::as_bool)
            == Some(true)
    {
        if let Some(provider) = api_key
            .preferences
            .get("__diagnostic_provider")
            .and_then(Value::as_str)
            .and_then(|name| snapshot.providers_by_name.get(name))
        {
            if !request_model.is_empty()
                && request_model.len() <= 256
                && !request_model.chars().any(char::is_control)
            {
                let mut probe = (**provider).clone();
                Arc::make_mut(&mut probe.models)
                    .entry(request_model.to_owned())
                    .or_insert_with(|| request_model.to_owned());
                matches.push(Arc::new(probe));
            }
        }
    }
    // First occurrence defines priority, including nested key and wildcard rules.
    // Sorting to deduplicate silently changes fixed_priority into name order.
    let mut seen = BTreeSet::new();
    matches.retain(|provider| seen.insert(provider.name.clone()));
    matches.retain(|provider| {
        crate::control::channels::temporary_allowed(provider, api_key)
            && provider_accepts_endpoint(provider, endpoint)
            && !provider.excluded_endpoints.iter().any(|excluded| {
                excluded
                    .trim_end_matches('/')
                    .eq_ignore_ascii_case(endpoint)
            })
            && provider_accepts_body(provider, request_body_bytes)
            && provider_accepts_request_type(provider, request_type)
            && provider_accepts_request_rules(
                provider,
                endpoint,
                request_model,
                reasoning_effort,
                request_type,
            )
    });
    Ok(matches)
}

pub(crate) fn weighted_provider_sequence(
    providers: &[Arc<Provider>],
    request_model: &str,
    weights: &Map<String, Value>,
    algorithm: &str,
) -> Vec<Arc<Provider>> {
    let weighted = providers
        .iter()
        .filter_map(|provider| {
            let exact = format!("{}/{request_model}", provider.name);
            let wildcard = format!("{}/*", provider.name);
            let weight = weights
                .get(&exact)
                .or_else(|| weights.get(&wildcard))
                .and_then(positive_weight)?;
            Some((provider.clone(), weight))
        })
        .collect::<Vec<_>>();
    if weighted.len() <= 1 {
        return providers.to_vec();
    }
    match algorithm {
        "weighted_round_robin" | "smart_round_robin" => smooth_weighted_sequence(&weighted),
        "lottery" => lottery_sequence(&weighted, scheduling_seed("lottery", request_model)),
        _ => weighted.into_iter().map(|(provider, _)| provider).collect(),
    }
}

pub(crate) fn positive_weight(value: &Value) -> Option<usize> {
    let weight = value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        .or_else(|| value.as_str().and_then(|value| value.parse::<u64>().ok()))?;
    usize::try_from(weight).ok().filter(|value| *value > 0)
}

pub(crate) fn smooth_weighted_sequence(weighted: &[(Arc<Provider>, usize)]) -> Vec<Arc<Provider>> {
    let total = weighted
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<usize>()
        .min(16_384);
    let mut current = vec![0i64; weighted.len()];
    let total_weight = weighted
        .iter()
        .map(|(_, weight)| i64::try_from(*weight).unwrap_or(i64::MAX / 4))
        .sum::<i64>()
        .max(1);
    let mut sequence = Vec::with_capacity(total);
    for _ in 0..total {
        let mut selected = 0usize;
        let mut selected_current = i64::MIN;
        let mut selected_configured = 1i64;
        for (index, (_, weight)) in weighted.iter().enumerate() {
            let configured = i64::try_from(*weight).unwrap_or(i64::MAX / 4);
            current[index] = current[index].saturating_add(configured);
            if selected_current == i64::MIN
                || current[index].saturating_mul(selected_configured)
                    > selected_current.saturating_mul(configured)
            {
                selected = index;
                selected_current = current[index];
                selected_configured = configured;
            }
        }
        sequence.push(weighted[selected].0.clone());
        current[selected] = current[selected].saturating_sub(total_weight);
    }
    sequence
}

pub(crate) fn lottery_sequence(
    weighted: &[(Arc<Provider>, usize)],
    mut seed: u64,
) -> Vec<Arc<Provider>> {
    let total = weighted
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<usize>()
        .min(16_384);
    let total_weight = weighted
        .iter()
        .map(|(_, weight)| *weight as u64)
        .sum::<u64>()
        .max(1);
    let mut sequence = Vec::with_capacity(total);
    for _ in 0..total {
        seed = xorshift(seed);
        let ticket = seed % total_weight;
        let mut cumulative = 0u64;
        let selected = weighted
            .iter()
            .find(|(_, weight)| {
                cumulative = cumulative.saturating_add(*weight as u64);
                ticket < cumulative
            })
            .map(|(provider, _)| provider)
            .unwrap_or(&weighted[0].0);
        sequence.push(selected.clone());
    }
    sequence
}

pub(crate) fn shuffle_indices(indices: &mut [usize], mut seed: u64) {
    for index in (1..indices.len()).rev() {
        seed = xorshift(seed);
        indices.swap(index, seed as usize % (index + 1));
    }
}

pub(crate) fn shuffle_providers(providers: &mut [Arc<Provider>], mut seed: u64) {
    for index in (1..providers.len()).rev() {
        seed = xorshift(seed);
        providers.swap(index, seed as usize % (index + 1));
    }
}

pub(crate) fn scheduling_seed(scope: &str, request_model: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(request_model.as_bytes());
    hasher.update(
        SCHEDULING_NONCE
            .fetch_add(1, Ordering::Relaxed)
            .to_le_bytes(),
    );
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"))
}

pub(crate) fn xorshift(mut value: u64) -> u64 {
    if value == 0 {
        value = 0x9e37_79b9_7f4a_7c15;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

pub(crate) fn api_key_retry_budget(api_key: &ApiKey, provider_count: usize) -> usize {
    let configured = api_key
        .preferences
        .get("AUTO_RETRY")
        .map(|value| match value {
            Value::Bool(enabled) => usize::from(*enabled),
            Value::Number(number) => number.as_u64().unwrap_or(0) as usize,
            Value::String(text) => text.trim().parse::<usize>().unwrap_or(1),
            _ => 1,
        })
        .unwrap_or(1);
    provider_count.saturating_add(configured)
}

pub(crate) fn compute_retry_count(providers: &[Arc<Provider>]) -> usize {
    if providers.is_empty() {
        return 0;
    }
    let retry = if providers.len() == 1 && providers[0].api_keys.len() > 1 {
        providers[0].api_keys.len()
    } else {
        providers
            .iter()
            .map(|provider| provider.api_keys.len())
            .sum::<usize>()
            .saturating_mul(2)
            .min(10)
    };
    providers.len().saturating_add(retry)
}
