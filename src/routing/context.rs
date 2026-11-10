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
}

impl RequestFacts {
    pub fn new(model: &str, endpoint: &str, body_bytes: u64, payload: Option<&Value>) -> Self {
        let object = payload.and_then(Value::as_object);
        Self {
            model: model.into(),
            endpoint: endpoint.trim_end_matches('/').into(),
            body_bytes,
            request_type: crate::routing::filters::request_type_for_endpoint(object, endpoint),
            reasoning_effort: object.and_then(crate::routing::filters::request_reasoning_effort),
            stream: payload
                .and_then(|p| p.get("stream"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
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
}

pub(crate) struct RequestContext {
    pub snapshot: Arc<Snapshot>,
    pub caller: Arc<ApiKey>,
    pub facts: RequestFacts,
    pub admission: AdmissionReceipt,
    pub route: ResolvedRoute,
}
