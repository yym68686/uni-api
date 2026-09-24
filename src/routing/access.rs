use crate::config::snapshot::ApiKey;
use crate::config::snapshot::Provider;
use crate::routing::filters::provider_accepts_endpoint;
use crate::routing::timeouts::model_preference;
use crate::routing::timeouts::model_timeout;
use crate::routing::timeouts::resolve_timeouts;
use crate::routing::timeouts::Timeouts;
use crate::routing::types::AuthContext;
use crate::routing::types::RouteResolutionError;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::Persistence;
use crate::upstream::responses::prepare::pydantic_bool;
use axum::http::{HeaderMap, StatusCode};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

impl GatewayRuntime {
    pub async fn models_for_headers(&self, headers: &HeaderMap) -> Result<Vec<String>, u16> {
        self.models_for_endpoint(headers, "all").await
    }

    // Read-only projection of the caller's configured conversational routes.
    // Do not schedule requests or consult transient cooldowns/usage here.
    pub(crate) async fn codex_models_for_headers(
        &self,
        headers: &HeaderMap,
    ) -> Result<Vec<String>, u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let base = self.base_snapshot().await.ok_or(503u16)?;
        let controls = self.channel_controls.read().await;
        let snapshot = controls.overlay(base);
        let key = snapshot.api_keys.get(&token).ok_or(403u16)?;
        let key_id = crate::routing::catalog::key_id(&token);
        let models = crate::routing::catalog::ordered_entries(&snapshot, key)
            .into_iter()
            .filter(|(provider, model)| {
                crate::control::channels::temporary_allowed(provider, key)
                    && provider_accepts_endpoint(provider, "/v1/responses")
                    && !provider.excluded_endpoints.iter().any(|endpoint| {
                        endpoint
                            .trim_end_matches('/')
                            .eq_ignore_ascii_case("/v1/responses")
                    })
                    && !controls.disabled(&key_id, model, &provider.name)
                    && crate::providers::codex::models::is_conversational_model(model)
                    && provider.models.get(model).is_some_and(|upstream| {
                        crate::providers::codex::models::is_conversational_model(upstream)
                    })
            })
            .map(|(_, model)| model)
            .collect::<BTreeSet<_>>();
        Ok(models.into_iter().collect())
    }

    pub(crate) async fn models_for_endpoint(
        &self,
        headers: &HeaderMap,
        endpoint: &str,
    ) -> Result<Vec<String>, u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let api_key = snapshot.api_keys.get(&token).ok_or(403u16)?;
        let allowed = |p: &Arc<Provider>| {
            crate::control::channels::temporary_allowed(p, api_key)
                && provider_accepts_endpoint(p, endpoint)
        };
        let mut models = BTreeSet::new();
        for rule in api_key.model_rules.iter() {
            if rule == "all" {
                for provider in snapshot.providers.iter().filter(|p| allowed(p)) {
                    models.extend(provider.models.keys().cloned());
                }
                continue;
            }
            if rule.starts_with('<') && rule.ends_with('>') {
                let model = rule[1..rule.len() - 1].to_owned();
                if snapshot
                    .providers
                    .iter()
                    .filter(|p| allowed(p))
                    .any(|provider| provider.models.contains_key(&model))
                {
                    models.insert(model);
                }
                continue;
            }
            if let Some((provider_name, model_rule)) = rule.split_once('/') {
                if let Some(provider) = snapshot
                    .providers_by_name
                    .get(provider_name)
                    .filter(|p| allowed(p))
                {
                    if model_rule == "*" {
                        models.extend(provider.models.keys().cloned());
                    } else if provider.models.contains_key(model_rule) {
                        models.insert(model_rule.to_owned());
                    }
                }
                continue;
            }
            if snapshot
                .providers
                .iter()
                .filter(|p| allowed(p))
                .any(|provider| provider.models.contains_key(rule))
            {
                models.insert(rule.clone());
            }
        }
        Ok(models.into_iter().collect())
    }

    pub(crate) async fn authorize(
        &self,
        headers: &HeaderMap,
    ) -> Result<AuthContext, RouteResolutionError> {
        let token = extract_api_key(headers).ok_or_else(|| RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Invalid or missing API Key".into(),
        })?;
        let snapshot = self.snapshot().await.ok_or_else(|| RouteResolutionError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Runtime configuration is not ready".into(),
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
        Ok(AuthContext {
            api_key,
            api_key_count: snapshot.api_keys.len(),
        })
    }

    pub(crate) async fn moderation_enabled(
        &self,
        headers: &HeaderMap,
    ) -> Result<bool, RouteResolutionError> {
        Ok(self
            .authorize(headers)
            .await?
            .api_key
            .preferences
            .get("ENABLE_MODERATION")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub(crate) async fn channel_catalog(
        &self,
        headers: &HeaderMap,
        endpoint: &str,
        stream: bool,
        selected_key_id: Option<&str>,
    ) -> Result<(Vec<Value>, String, String), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        let selected_id = selected_key_id.unwrap_or_default();
        let entries = crate::routing::catalog::entries(&snapshot, caller, Some(selected_id))?;
        let cooldowns = self.scheduling.channel_cooldowns.lock().await.clone();
        let now = tokio::time::Instant::now();
        let controls = self.channel_controls.read().await.routing_rules();
        let mut rows: Vec<Value> = entries.into_iter().filter_map(|(provider, model)| {
            if !provider_accepts_endpoint(&provider, endpoint) || (provider.engine.eq_ignore_ascii_case("typesafe") && stream) || provider.excluded_endpoints.iter().any(|v| v.trim_end_matches('/').eq_ignore_ascii_case(endpoint)) {
                return None;
            }
            let upstream = provider.models.get(&model)?;
            let route_cooling = cooldowns.get(&(provider.name.to_string(), upstream.clone())).is_some_and(|until| *until > now);
            let (eligible, reason) = if controls.disabled(selected_id,&model,&provider.name) {
                (false,"temporarily_disabled")
            } else if provider.api_keys.is_empty() && provider.client_email.is_none() {
                (false, "no_provider_key")
            } else if route_cooling { (false, "channel_cooldown") } else { (true, "eligible") };
            Some(json!({"provider":provider.name.as_ref(),"model":model,"upstream_model":upstream,"engine":provider.engine.as_ref(),"endpoint":endpoint,"stream":stream,"eligible":eligible,"reason":reason}))
        }).collect();
        rows.sort_by_key(|row| {
            controls
                .order(selected_id, row["model"].as_str().unwrap_or_default())
                .and_then(|order| {
                    order
                        .iter()
                        .position(|p| Some(p.as_str()) == row["provider"].as_str())
                })
                .unwrap_or(usize::MAX)
        });
        Ok((rows, snapshot.revision.to_string(), selected_id.to_owned()))
    }

    pub(crate) async fn authorize_catalog(&self, headers: &HeaderMap) -> Result<(), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::routing::catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        Ok(())
    }

    pub(crate) async fn balance_provider(
        &self,
        headers: &HeaderMap,
        name: &str,
    ) -> Result<(Arc<Provider>, Option<String>), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::routing::catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        let provider = snapshot
            .providers_by_name
            .get(name)
            .cloned()
            .ok_or(404u16)?;
        let proxy = preference_string(&provider.preferences, "proxy")
            .or_else(|| preference_string(&snapshot.preferences, "proxy"));
        Ok((provider, proxy))
    }

    pub(crate) async fn api_key_catalog(&self, headers: &HeaderMap) -> Result<Value, u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::routing::catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        Ok(
            json!({"data":crate::routing::catalog::keys(&snapshot, caller),"snapshot_revision":snapshot.revision.as_ref(),"can_inspect_all":true}),
        )
    }

    pub(crate) async fn prices_for_model(&self, model: &str) -> (f64, f64) {
        self.snapshot()
            .await
            .map(|snapshot| model_prices(&snapshot.preferences, model))
            .unwrap_or((0.3, 1.0))
    }

    pub(crate) async fn keepalive_interval(
        &self,
        provider: &Provider,
        request_model: &str,
        original_model: &str,
    ) -> Option<Duration> {
        let snapshot = self.snapshot().await?;
        let interval = model_preference(
            provider,
            &snapshot.preferences,
            request_model,
            original_model,
            "keepalive_interval",
        )
        .unwrap_or(99999.0);
        let timeout = model_timeout(
            provider,
            &snapshot.preferences,
            request_model,
            original_model,
        );
        (interval.is_finite() && interval > 0.0 && interval <= timeout)
            .then(|| Duration::from_secs_f64(interval))
    }

    pub(crate) async fn auto_retry_enabled(&self, headers: &HeaderMap) -> bool {
        self.auto_retry_budget(headers).await > 0
    }

    pub(crate) async fn auto_retry_budget(&self, headers: &HeaderMap) -> usize {
        let Some(snapshot) = self.snapshot().await else {
            return 1;
        };
        let Some(token) = extract_api_key(headers) else {
            return 1;
        };
        let Some(value) = snapshot
            .api_keys
            .get(&token)
            .and_then(|key| key.preferences.get("AUTO_RETRY"))
        else {
            return 1;
        };
        match value {
            Value::Bool(enabled) => usize::from(*enabled),
            Value::Number(number) => number.as_u64().unwrap_or(0).min(100) as usize,
            Value::String(text) => text
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| usize::from(pydantic_bool(value).unwrap_or(true)))
                .min(100),
            _ => 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn generic_timeouts(
        &self,
        headers: &HeaderMap,
        provider: &Provider,
        request_model: &str,
        original_model: &str,
        engine: &str,
        stream: bool,
        endpoint: &str,
        method: &str,
    ) -> Timeouts {
        let Some(snapshot) = self.snapshot().await else {
            return Timeouts::default();
        };
        let role = extract_api_key(headers)
            .and_then(|token| snapshot.api_keys.get(&token).cloned())
            .map(|key| key.role.to_string())
            .unwrap_or_default();
        resolve_timeouts(
            &snapshot,
            provider,
            request_model,
            original_model,
            engine,
            stream,
            None,
            &role,
            endpoint,
            method,
        )
    }

    pub async fn paid_api_key_states(&self, persistence: &Persistence) -> Value {
        let Some(snapshot) = self.snapshot().await else {
            return json!({});
        };
        let mut states = Map::new();
        for item in snapshot
            .api_config
            .get("api_keys")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(token) = item.get("api").and_then(Value::as_str) else {
                continue;
            };
            let Some(credits) = item.pointer("/preferences/credits").and_then(Value::as_f64) else {
                continue;
            };
            let created_at = item
                .pointer("/preferences/created_at")
                .and_then(Value::as_str)
                .and_then(parse_config_datetime)
                .unwrap_or_else(|| unix_seconds_i64().saturating_sub(30 * 86_400));
            let total_cost = persistence
                .total_cost(token, created_at)
                .await
                .unwrap_or(0.0);
            let all_tokens_info = persistence
                .token_usage(Some(token), None, Some(created_at), None)
                .await
                .ok()
                .and_then(|value| value.get("usage").cloned())
                .unwrap_or_else(|| json!([]));
            states.insert(
                token.to_owned(),
                json!({
                    "credits": credits,
                    "created_at": created_at,
                    "all_tokens_info": all_tokens_info,
                    "total_cost": total_cost,
                    "enabled": credits == -1.0 || total_cost <= credits,
                }),
            );
        }
        Value::Object(states)
    }

    pub(crate) async fn ensure_paid_balance(
        &self,
        persistence: &Persistence,
        api_key: &ApiKey,
    ) -> Result<(), RouteResolutionError> {
        if persistence.disabled() {
            return Ok(());
        }
        let Some(credits) = api_key.preferences.get("credits").and_then(Value::as_f64) else {
            return Ok(());
        };
        if credits == -1.0 {
            return Ok(());
        }
        let created_at = api_key
            .preferences
            .get("created_at")
            .and_then(Value::as_str)
            .and_then(parse_config_datetime)
            .unwrap_or_else(|| unix_seconds_i64().saturating_sub(30 * 86_400));
        let total_cost = persistence
            .total_cost(api_key.token.as_ref(), created_at)
            .await
            .map_err(|error| RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: format!("Unable to verify API-key balance: {error}"),
            })?;
        if total_cost > credits {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Balance is insufficient, please check your account.".into(),
            });
        }
        Ok(())
    }
}

pub(crate) fn model_prices(preferences: &Map<String, Value>, model: &str) -> (f64, f64) {
    let prices = preferences.get("model_price").and_then(Value::as_object);
    let encoded = prices
        .and_then(|prices| {
            prices
                .iter()
                .find(|(prefix, _)| {
                    prefix.as_str() != "default" && model.starts_with(prefix.as_str())
                })
                .map(|(_, value)| value)
                .or_else(|| prices.get("default"))
        })
        .and_then(|value| value.as_str())
        .unwrap_or("0.3,1");
    let mut parts = encoded.split(',').map(str::trim);
    let prompt = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.3);
    let completion = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);
    (prompt, completion)
}

pub(crate) fn unix_seconds_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

pub(crate) fn parse_config_datetime(value: &str) -> Option<i64> {
    if let Ok(value) = value.parse::<f64>() {
        return value.is_finite().then_some(value as i64);
    }
    let value = value.trim();
    let (date, raw_time) = value.split_once('T').or_else(|| value.split_once(' '))?;
    let (time, offset_seconds) = if let Some(time) = raw_time.strip_suffix('Z') {
        (time, 0i64)
    } else if let Some(index) = raw_time
        .char_indices()
        .skip(1)
        .filter(|(_, value)| matches!(value, '+' | '-'))
        .map(|(index, _)| index)
        .fold(None, |_, index| Some(index))
    {
        let (time, offset) = raw_time.split_at(index);
        let sign = if offset.starts_with('-') { -1 } else { 1 };
        let mut parts = offset[1..].split(':');
        let hours = parts.next()?.parse::<i64>().ok()?;
        let minutes = parts.next().unwrap_or("0").parse::<i64>().ok()?;
        (time, sign * (hours * 3600 + minutes * 60))
    } else {
        (raw_time, 0i64)
    };
    let mut date = date.split('-').map(|value| value.parse::<i64>().ok());
    let (year, month, day) = (date.next()??, date.next()??, date.next()??);
    let mut time = time.split(':');
    let hour = time.next()?.parse::<i64>().ok()?;
    let minute = time.next()?.parse::<i64>().ok()?;
    let second = time.next()?.split('.').next()?.parse::<i64>().ok()?;
    Some(
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second
            - offset_seconds,
    )
}

pub(crate) fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub(crate) fn preference_string(preferences: &Map<String, Value>, key: &str) -> Option<String> {
    preferences
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(crate) fn preference_f64(preferences: &Map<String, Value>, key: &str) -> Option<f64> {
    preferences.get(key).and_then(Value::as_f64)
}

pub(crate) fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(token) = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
    {
        if !token.is_empty() {
            return Some(token.to_owned());
        }
    }
    let authorization = headers.get("authorization")?.to_str().ok()?;
    authorization
        .split_once(' ')
        .map(|(_, token)| token.trim())
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}
