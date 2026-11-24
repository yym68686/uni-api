use crate::api::platform::{json_response, require_admin};
use crate::routing::attempt_policy::ResolvedAttemptPolicy;
use crate::routing::service_tier::{self as tiers, Tier};
use crate::runtime::context::AppState;
use crate::transport::http::json_error;
use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::Duration;

pub(crate) const PATH: &str = "/v1/channel-settings/service-tier-probe";
const BUDGET: Duration = Duration::from_secs(40);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    provider: String,
    model: String,
    tier: Tier,
    #[serde(default)]
    verify_only: bool,
    #[serde(default)]
    expected_scope: String,
    candidate: Option<Candidate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    base_url: String,
    api_key: String,
    models: BTreeMap<String, String>,
}

fn valid_model(v: &str) -> bool {
    !v.is_empty() && v.len() <= 256 && !v.chars().any(char::is_control)
}

fn preview_candidate(
    snapshot: &crate::config::snapshot::Snapshot,
    provider: &str,
    c: Candidate,
) -> Option<Value> {
    if !crate::control::settings::valid_created_name(provider)
        || snapshot.providers_by_name.contains_key(provider)
        || c.base_url.is_empty()
        || c.base_url.len() > 8192
        || c.base_url.chars().any(char::is_control)
        || c.api_key.is_empty()
        || c.api_key.len() > 8192
        || c.api_key.contains([',', '\r', '\n'])
        || c.models.is_empty()
        || c.models.len() > 128
        || !c
            .models
            .iter()
            .all(|(public, upstream)| valid_model(public) && valid_model(upstream))
    {
        return None;
    }
    let models: Vec<_> = c
        .models
        .iter()
        .map(|(public, upstream)| json!({upstream:public}))
        .collect();
    let raw = json!({"provider":provider,"base_url":c.base_url,"api":[c.api_key],"model":models});
    let compiled = crate::config::compiler::compile_provider(&raw)?;
    let provider = crate::config::snapshot::runtime_provider(
        serde_json::from_value(compiled).ok()?,
        Arc::new(AtomicUsize::new(0)),
    );
    let key = provider.api_keys.first()?;
    let mut out = serde_json::Map::new();
    for model in provider.models.keys() {
        if let Some(url) = tiers::endpoint(&provider, model) {
            out.insert(
                model.clone(),
                json!({
                    "binding": tiers::binding(snapshot,&provider,model,&url,key),
                    "compatibility": tiers::compatibility(snapshot,&provider,model,&url,key),
                }),
            );
        }
    }
    Some(json!({"models":out}))
}

pub(crate) async fn handle(state: &AppState, request: Request) -> Response<Body> {
    if let Err(response) = require_admin(state, request.headers()).await {
        return response;
    }
    let Ok(raw) = to_bytes(request.into_body(), 100 * 1024).await else {
        return json_error(StatusCode::PAYLOAD_TOO_LARGE, "Probe request too large");
    };
    let Ok(input) = serde_json::from_slice::<Input>(&raw) else {
        return json_error(StatusCode::BAD_REQUEST, "Invalid service tier probe");
    };
    if [&input.provider, &input.model]
        .iter()
        .any(|v| !valid_model(v))
    {
        return json_error(StatusCode::BAD_REQUEST, "Invalid service tier probe target");
    }
    let Some(snapshot) = state.runtime.snapshot().await else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Runtime configuration is not ready",
        );
    };
    if let Some(candidate) = input.candidate {
        if !input.verify_only || !input.expected_scope.is_empty() {
            return json_error(
                StatusCode::BAD_REQUEST,
                "A candidate can only be verified locally",
            );
        }
        return match preview_candidate(&snapshot, &input.provider, candidate) {
            Some(value) => json_response(StatusCode::OK, value),
            None => json_error(StatusCode::CONFLICT, "Candidate could not be verified"),
        };
    }
    let Some(provider) = snapshot.providers_by_name.get(&input.provider) else {
        return json_error(StatusCode::NOT_FOUND, "Provider not found");
    };
    let scope = tiers::inventory_scope(&snapshot, provider);
    if !input.expected_scope.is_empty() && input.expected_scope != scope {
        return json_error(StatusCode::CONFLICT, "Channel configuration changed");
    }
    let provider = crate::config::model_protocol::effective(provider.clone(), &input.model);
    let checked_at = crate::runtime::clock::unix_seconds();
    let unsupported_target = || {
        json_response(
            StatusCode::OK,
            json!({
                "status":"not_applicable","checked_at":checked_at,"scope":scope,
            }),
        )
    };
    let Some(upstream_model) = provider.models.get(&input.model) else {
        return unsupported_target();
    };
    let Some(url) = tiers::endpoint(&provider, &input.model) else {
        return unsupported_target();
    };
    // A single explicitly requested probe uses the first configured credential.
    // Never silently try another credential or transfer evidence to another key.
    let Some(key) = provider
        .api_keys
        .first()
        .filter(|key| !key.contains(',') && !key.is_empty())
    else {
        return unsupported_target();
    };
    let binding = tiers::binding(&snapshot, &provider, &input.model, &url, key);
    let compatibility = tiers::compatibility(&snapshot, &provider, &input.model, &url, key);
    if input.verify_only {
        return json_response(
            StatusCode::OK,
            json!({"status":"ready","binding":binding,"scope":scope,"compatibility":compatibility,"upstream_model":upstream_model}),
        );
    }
    let stream =
        crate::providers::stream_mode::stream_only(&provider, &input.model, upstream_model)
            || crate::providers::responses::provider_stream_override(&provider)
                .unwrap_or(provider.engine.as_ref() == "codex");
    let request_id = crate::observability::context::request_id(&HeaderMap::new());
    let Ok(headers) = crate::providers::responses::build_headers(
        &HeaderMap::new(),
        &provider,
        key,
        &provider.engine,
        stream,
        &request_id,
        &request_id,
    ) else {
        return unsupported_target();
    };
    // Do not run arbitrary body overrides on this billable diagnostic. The only
    // input is the fixed minimal prompt, with no tools or background work.
    let mut body = json!({
        "model":upstream_model,"input":"Reply OK.","service_tier":input.tier.wire(),
        "store":false,"stream":stream,
    });
    if provider.engine.as_ref() == "codex" {
        body["instructions"] = json!("");
        body["input"] =
            json!([{"role":"user","content":[{"type":"input_text","text":"Reply OK."}]}]);
    }
    let mut client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        // A fresh HTTP/1 client avoids pooled and HTTP/2 retries.
        .http1_only()
        .connect_timeout(Duration::from_secs(8));
    if let Some(proxy) = ResolvedAttemptPolicy::proxy(&snapshot, &provider) {
        let Ok(proxy) = reqwest::Proxy::all(proxy) else {
            return unsupported_target();
        };
        client = client.proxy(proxy);
    }
    let Ok(client) = client.build() else {
        return unsupported_target();
    };
    let result = tokio::time::timeout(BUDGET, async {
        let mut req = client.post(url).body(body.to_string());
        for (name, value) in headers {
            req = req.header(name, value);
        }
        let response = req.send().await.ok()?;
        Some(evaluate(response, upstream_model, input.tier).await)
    })
    .await
    .ok()
    .flatten()
    .unwrap_or(("inconclusive", None));
    json_response(
        StatusCode::OK,
        json!({
            "status":result.0, "observed":result.1, "binding":binding, "checked_at":checked_at, "scope":scope,
            "compatibility":compatibility, "upstream_model":upstream_model,
        }),
    )
}

async fn evaluate(
    mut response: reqwest::Response,
    model: &str,
    tier: Tier,
) -> (&'static str, Option<&'static str>) {
    let status = response.status();
    if status == StatusCode::OK
        && response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"))
            })
    {
        return match crate::protocols::responses::collect::collect(
            response,
            tokio::time::Instant::now(),
            Some(Duration::from_secs(15)),
            false,
            false,
        )
        .await
        {
            Ok(value) => completed(&value.payload, model, tier),
            Err(error) => rejected(&error.outcome["payload"], error.status),
        };
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) if body.len() + chunk.len() <= 256 * 1024 => {
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            _ => return ("inconclusive", None),
        }
    }
    let Ok(value) = serde_json::from_slice::<Value>(&body) else {
        return ("inconclusive", None);
    };
    if status == StatusCode::OK {
        completed(&value, model, tier)
    } else {
        rejected(&value, status.as_u16())
    }
}

fn completed(value: &Value, model: &str, tier: Tier) -> (&'static str, Option<&'static str>) {
    if value["status"] != "completed"
        || value["model"] != model
        || !value["output"].is_array()
        || !value["error"].is_null()
        || value["id"].as_str().is_none_or(str::is_empty)
    {
        return ("inconclusive", None);
    }
    let observed = match value.get("service_tier").and_then(Value::as_str) {
        Some("fast") => "fast",
        Some("priority") => "priority",
        Some("ultrafast") => "ultrafast",
        Some("default") => "default",
        Some("flex") => "flex",
        Some("scale") => "scale",
        _ => return ("inconclusive", None),
    };
    (
        if tier.served(observed) {
            "supported"
        } else {
            "not_honored"
        },
        Some(observed),
    )
}

fn rejected(value: &Value, status: u16) -> (&'static str, Option<&'static str>) {
    let err = value
        .get("error")
        .or_else(|| value.pointer("/response/error"))
        .unwrap_or(value);
    let explicitly_denied = err["param"] == "service_tier"
        && (matches!(
            err["code"].as_str(),
            Some(
                "unsupported_service_tier"
                    | "service_tier_not_supported"
                    | "service_tier_not_allowed"
            )
        ) || err["message"].as_str().is_some_and(|s| {
            s.contains("The requested service tier is not allowed for this project.")
        }));
    if matches!(status, 400 | 422) && explicitly_denied {
        ("unsupported", None)
    } else {
        ("inconclusive", None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_completed_matching_model_and_explicit_served_tier_can_verify_support() {
        let mut value = json!({"id":"resp_test","output":[],"status":"completed","model":"wire","service_tier":"priority"});
        assert_eq!(
            completed(&value, "wire", Tier::Fast),
            ("supported", Some("priority"))
        );
        assert_eq!(
            completed(&value, "wire", Tier::Ultrafast),
            ("not_honored", Some("priority"))
        );
        value["service_tier"] = json!("default");
        assert_eq!(
            completed(&value, "wire", Tier::Ultrafast),
            ("not_honored", Some("default"))
        );
        value["service_tier"] = json!("ultrafast");
        assert_eq!(
            completed(&value, "wire", Tier::Ultrafast),
            ("supported", Some("ultrafast"))
        );
        assert_eq!(
            completed(&value, "other", Tier::Ultrafast),
            ("inconclusive", None)
        );
        value["status"] = json!("incomplete");
        assert_eq!(
            completed(&value, "wire", Tier::Ultrafast),
            ("inconclusive", None)
        );
    }
}
