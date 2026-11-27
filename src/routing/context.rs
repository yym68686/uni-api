//! Immutable request inputs and the results of request-scoped side effects.
use crate::config::snapshot::{ApiKey, Snapshot};
use crate::routing::types::ResolvedRoute;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct RequestFacts {
    pub model: String,
    pub endpoint: String,
    pub body_bytes: u64,
    pub request_type: Option<&'static str>,
    pub reasoning_effort: Option<String>,
    pub stream: bool,
    pub service_tier: Option<crate::routing::service_tier::Tier>,
    pub requested_service_tier: Option<&'static str>,
    pub interactions_compat_error: Option<String>,
}

impl RequestFacts {
    pub(crate) fn log_request_type(&self) -> Option<&'static str> {
        Some(if self.request_type == Some("compaction") {
            "compaction"
        } else {
            "standard"
        })
    }

    pub(crate) fn log_service_tier(&self) -> Option<&'static str> {
        self.requested_service_tier
    }

    pub fn new(model: &str, endpoint: &str, body_bytes: u64, payload: Option<&Value>) -> Self {
        let object = payload.and_then(Value::as_object);
        Self {
            model: model.into(),
            endpoint: endpoint.trim_end_matches('/').into(),
            body_bytes,
            request_type: crate::routing::filters::request_type_for_endpoint(object, endpoint),
            reasoning_effort: object.and_then(crate::routing::filters::request_reasoning_effort),
            service_tier: crate::routing::service_tier::Tier::requested(endpoint, payload),
            requested_service_tier: requested_log_tier(endpoint, payload),
            interactions_compat_error: payload
                .filter(|_| crate::protocols::interactions::is_route(endpoint))
                .and_then(|body| crate::protocols::interactions::compatibility(body).err()),
            stream: payload
                .and_then(|p| p.get("stream"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }
}

// Logging is deliberately independent of routing eligibility: a requested
// tier is not proof of upstream execution or of the charged price.
fn requested_log_tier(endpoint: &str, payload: Option<&Value>) -> Option<&'static str> {
    let object = payload?.as_object()?;
    let field = if endpoint.trim_end_matches('/') == "/v1/messages" {
        "speed"
    } else {
        "service_tier"
    };
    match object.get(field) {
        None | Some(Value::Null) => Some("default"),
        Some(value) => match (field, value.as_str()?) {
            ("speed", "standard") | ("service_tier", "default") => Some("default"),
            ("speed", "fast") | ("service_tier", "fast" | "priority") => Some("fast"),
            ("service_tier", "ultrafast") => Some("ultrafast"),
            ("service_tier", "flex") => Some("flex"),
            ("service_tier", "scale") => Some("scale"),
            ("service_tier", "auto") => Some("auto"),
            _ => None,
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionReceipt {
    Request,
    Internal,
}

/// This is an in-process extension, never a caller-controlled HTTP flag.
#[derive(Clone, Copy)]
pub(crate) struct ModerationReceipt;

#[derive(Clone)]
pub(crate) struct PreparedPayload {
    pub payload: Arc<Value>,
    pub observation: crate::transport::spool::SpoolObservation,
    pub reservation: Arc<crate::runtime::resources::MemoryReservation>,
}

pub(crate) struct RequestContext {
    pub snapshot: Arc<Snapshot>,
    pub caller: Arc<ApiKey>,
    pub facts: RequestFacts,
    pub admission: AdmissionReceipt,
    pub route: ResolvedRoute,
}
