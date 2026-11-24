//! Only an exact, recent Responses HTTP probe can qualify an upstream credential.
use crate::config::snapshot::{Provider, Snapshot};
use crate::routing::attempt_policy::ResolvedAttemptPolicy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(crate) const ENDPOINT: &str = "/v1/responses";
pub(crate) const SWITCH: &str = "service_tier_routing";
pub(crate) const EVIDENCE: &str = "service_tier_evidence";
pub(crate) const DISABLED: &str = "service_tier_disabled";
pub(crate) const ROUTED: &str = "__verified_service_tier";
pub(crate) const TTL: i64 = 7 * 24 * 60 * 60;
pub(crate) const UNAVAILABLE: &str =
    "No eligible channel has verified support for the requested service_tier";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Tier {
    Fast,
    Ultrafast,
}

impl Tier {
    pub(crate) fn requested(endpoint: &str, payload: Option<&Value>) -> Option<Self> {
        if endpoint.trim_end_matches('/') != ENDPOINT {
            return None;
        }
        match payload?.get("service_tier")?.as_str()? {
            "fast" | "priority" => Some(Self::Fast),
            "ultrafast" => Some(Self::Ultrafast),
            _ => None,
        }
    }

    pub(crate) fn wire(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Ultrafast => "ultrafast",
        }
    }

    pub(crate) fn served(self, actual: &str) -> bool {
        actual == self.wire() || (self == Self::Fast && actual == "priority")
    }
}

pub(crate) fn enabled(snapshot: &Snapshot) -> bool {
    snapshot.preferences.get(SWITCH).and_then(Value::as_bool) == Some(true)
}

pub(crate) fn endpoint(provider: &Provider, model: &str) -> Option<String> {
    let engine = provider.engine.as_ref();
    if !matches!(engine, "gpt" | "codex")
        || provider
            .excluded_endpoints
            .iter()
            .any(|e| e.trim_end_matches('/').eq_ignore_ascii_case(ENDPOINT))
        || !crate::routing::filters::provider_accepts_request_type(provider, None)
        || provider
            .preferences
            .get("passthrough_request_headers")
            .is_some_and(|v| !crate::config::snapshot::endpoint_values(v).is_empty())
    {
        return None;
    }
    if provider
        .preferences
        .get("headers")
        .and_then(Value::as_object)
        .is_some_and(|headers| {
            headers.keys().any(|name| {
                matches!(
                    name.to_ascii_lowercase().as_str(),
                    "authorization"
                        | "host"
                        | "cookie"
                        | "proxy-authorization"
                        | "x-api-key"
                        | "api-key"
                )
            })
        })
    {
        return None;
    }
    // A model rewrite would invalidate the configured public-to-wire mapping.
    if provider
        .preferences
        .get("post_body_parameter_overrides")
        .is_some_and(|v| {
            v.get("model").is_some() || v.get(model).is_some_and(|p| p.get("model").is_some())
        })
    {
        return None;
    }
    let native =
        crate::providers::responses::normalize_upstream_url(&provider.base_url, engine, false);
    let url = url::Url::parse(&native).ok()?;
    let generic =
        crate::providers::endpoints::replace_known_endpoint(&provider.base_url, ENDPOINT).ok()?;
    // Both Rust execution paths must reach the same upstream address.
    if !matches!(url.scheme(), "http" | "https")
        || !url.has_host()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.query().is_some()
        || !url.path().ends_with("/responses")
        || url.as_str() != generic
    {
        return None;
    }
    Some(url.into())
}

pub(crate) fn binding(
    snapshot: &Snapshot,
    provider: &Provider,
    model: &str,
    url: &str,
    key: &str,
) -> String {
    let p = &provider.preferences;
    let scope = json!([
        "responses-http-v1",
        provider.name.as_ref(),
        url,
        provider.engine.as_ref(),
        model,
        provider.models.get(model),
        ResolvedAttemptPolicy::proxy(snapshot, provider),
        p.get("headers"),
        p.get("upstream_stream_mode"),
        p.get("post_body_parameter_overrides")
    ]);
    let mut hash = Sha256::new();
    hash.update(scope.to_string().as_bytes());
    hash.update([0]);
    hash.update(key.as_bytes());
    format!("{:x}", hash.finalize())
}

// Inventory fence for presenting previous results on the current channel.
pub(crate) fn inventory_scope(snapshot: &Snapshot, provider: &Provider) -> String {
    let p = &provider.preferences;
    let wire = json!([
        "responses-http-inventory-v1",
        provider.name.as_ref(),
        provider.base_url.as_ref(),
        provider.engine.as_ref(),
        serde_json::to_value(provider.models.as_ref()).unwrap_or_default(),
        provider.api_keys.first(),
        ResolvedAttemptPolicy::proxy(snapshot, provider),
        p.get("__model_upstream_protocols"),
        p.get("headers"),
        p.get("passthrough_request_headers"),
        p.get("upstream_stream_mode"),
        p.get("post_body_parameter_overrides"),
        provider.excluded_endpoints.as_ref(),
        provider.only_request_types.as_ref(),
        provider.excluded_request_types.as_ref(),
    ]);
    format!("{:x}", Sha256::digest(wire.to_string().as_bytes()))
}

pub(crate) fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}

pub(crate) fn valid_evidence(value: &Value) -> bool {
    value.as_object().is_some_and(|m| {
        m.len() <= 2048
            && m.iter().all(|(id, v)| {
                is_digest(id)
                    && v.as_object().is_some_and(|tiers| {
                        tiers.iter().all(|(tier, ts)| {
                            matches!(tier.as_str(), "fast" | "ultrafast")
                                && ts.as_i64().is_some_and(|time| time > 0)
                        })
                    })
            })
    })
}

pub(crate) fn valid_disabled(value: &Value) -> bool {
    value.as_object().is_some_and(|tiers| {
        tiers.iter().all(|(tier, disabled)| {
            matches!(tier.as_str(), "fast" | "ultrafast") && disabled.is_boolean()
        })
    })
}

pub(crate) fn disabled(provider: &Provider, tier: Tier) -> bool {
    provider
        .preferences
        .get(DISABLED)
        .and_then(|v| v.get(tier.wire()))
        .and_then(Value::as_bool)
        == Some(true)
}

pub(crate) fn fresh(time: i64, now: i64) -> bool {
    time > 0 && time <= now && now.saturating_sub(time) < TTL
}

pub(crate) fn restrict(
    snapshot: &Snapshot,
    model: &str,
    mut provider: Arc<Provider>,
    tier: Tier,
    now: i64,
) -> Option<Arc<Provider>> {
    if disabled(&provider, tier) {
        return None;
    }
    let keys = qualified_keys(snapshot, &provider, model, tier, now)?;
    let selected = Arc::make_mut(&mut provider);
    selected.api_keys = Arc::new(keys);
    Arc::make_mut(&mut selected.preferences).insert(ROUTED.into(), json!(tier.wire()));
    Some(provider)
}

fn qualified_keys(
    snapshot: &Snapshot,
    provider: &Provider,
    model: &str,
    tier: Tier,
    now: i64,
) -> Option<Vec<String>> {
    let evidence = provider.preferences.get(EVIDENCE)?;
    let url = endpoint(provider, model)?;
    let keys: Vec<_> = provider
        .api_keys
        .iter()
        .filter(|key| {
            !key.is_empty()
                && !key.contains(',')
                && evidence
                    .get(binding(snapshot, provider, model, &url, key))
                    .and_then(|v| v.get(tier.wire()))
                    .and_then(Value::as_i64)
                    .is_some_and(|time| fresh(time, now))
        })
        .cloned()
        .collect();
    (!keys.is_empty()).then_some(keys)
}

// Return only route state, not bindings or raw credentials. This describes
// this provider's configuration, independent of the global routing switch.
pub(crate) fn routing_state(snapshot: &Snapshot, provider: &Arc<Provider>, now: i64) -> Value {
    let mut result = serde_json::Map::new();
    let evidence = provider.preferences.get(EVIDENCE);
    for tier in [Tier::Fast, Tier::Ultrafast] {
        let has_evidence = evidence.is_some_and(|e| {
            e.as_object().is_some_and(|o| {
                o.values().any(|v| {
                    v.get(tier.wire())
                        .and_then(Value::as_i64)
                        .is_some_and(|at| fresh(at, now))
                })
            })
        }) && provider.models.keys().any(|model| {
            qualified_keys(
                snapshot,
                &crate::config::model_protocol::effective(provider.clone(), model),
                model,
                tier,
                now,
            )
            .is_some()
        });
        let state = match (disabled(provider, tier), has_evidence) {
            (true, true) => "disabled_verified",
            (true, false) => "disabled",
            (false, true) => "enabled",
            (false, false) => "unconfigured",
        };
        result.insert(tier.wire().into(), json!(state));
    }
    Value::Object(result)
}

// Portable receipts are issued only for plain Responses channels. Transfer
// still requires the same authenticated destination, wire model, protocol,
// proxy and streaming behavior. Provider names and public aliases are not
// wire inputs; routing evidence itself remains bound to those names.
pub(crate) fn compatibility(
    snapshot: &Snapshot,
    provider: &Provider,
    model: &str,
    url: &str,
    key: &str,
) -> Option<String> {
    if [
        "headers",
        "post_body_parameter_overrides",
        "passthrough_request_headers",
    ]
    .iter()
    .any(|field| provider.preferences.get(*field).is_some())
    {
        return None;
    }
    let upstream = provider.models.get(model)?;
    let stream = crate::providers::stream_mode::stream_only(provider, model, upstream)
        || crate::providers::responses::provider_stream_override(provider)
            .unwrap_or(provider.engine.as_ref() == "codex");
    let wire = json!([
        "responses-portable-v1",
        url,
        provider.engine.as_ref(),
        upstream,
        ResolvedAttemptPolicy::proxy(snapshot, provider),
        stream
    ]);
    let mut hash = Sha256::new();
    hash.update(wire.to_string().as_bytes());
    hash.update([0]);
    hash.update(key.as_bytes());
    Some(format!("{:x}", hash.finalize()))
}

pub(crate) fn rejection(error: &super::types::RouteResolutionError) -> Option<Value> {
    (error.status == axum::http::StatusCode::SERVICE_UNAVAILABLE && error.message == UNAVAILABLE)
        .then(|| json!({"error":{"message":error.message,"type":"server_error","param":"service_tier","code":"service_tier_unavailable"}}))
}
