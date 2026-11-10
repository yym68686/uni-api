//! Common final facts; transport status is never inferred from semantic status.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TerminalKind {
    Completed,
    Incomplete,
    SemanticFailure,
    ProtocolError,
    TransportError,
    DownstreamDisconnected,
    HedgeCancelled,
    Rejected,
    Unknown,
}

impl TerminalKind {
    pub fn parse(kind: &str) -> Self {
        match kind {
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "semantic_error" | "semantic_failure" | "failed" => Self::SemanticFailure,
            "protocol_error" => Self::ProtocolError,
            "transport_error" | "timeout" => Self::TransportError,
            "cancelled" | "downstream_disconnected" => Self::DownstreamDisconnected,
            "hedge_cancelled" => Self::HedgeCancelled,
            "native_rejection" | "http_error" => Self::Rejected,
            _ => Self::Unknown,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::SemanticFailure => "semantic_failure",
            Self::ProtocolError => "protocol_error",
            Self::TransportError => "transport_error",
            Self::DownstreamDisconnected => "downstream_disconnected",
            Self::HedgeCancelled => "hedge_cancelled",
            Self::Rejected => "rejected",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FinalFacts {
    pub terminal: TerminalKind,
    pub response_completed: Option<bool>,
    pub business_committed: Option<bool>,
    pub upstream_http_status: Option<u16>,
    pub downstream_http_status: Option<u16>,
    pub semantic_status: u16,
}

impl FinalFacts {
    pub fn buffered(
        value: Option<&Value>,
        status: u16,
        downstream_status: u16,
        _stream: bool,
    ) -> Self {
        let terminal = value
            .and_then(|v| v["kind"].as_str())
            .map(TerminalKind::parse)
            .unwrap_or(TerminalKind::Unknown);
        Self {
            terminal,
            response_completed: value.map(|_| terminal == TerminalKind::Completed),
            business_committed: None,
            upstream_http_status: None,
            downstream_http_status: Some(downstream_status),
            semantic_status: status,
        }
    }

    pub fn exhausted(status: u16) -> Self {
        Self {
            terminal: TerminalKind::Unknown,
            response_completed: None,
            business_committed: None,
            upstream_http_status: None,
            downstream_http_status: None,
            semantic_status: status,
        }
    }

    pub fn native(value: &Value, semantic_status: u16, responses_protocol: bool) -> Self {
        let terminal = TerminalKind::parse(value["kind"].as_str().unwrap_or("unknown"));
        Self {
            terminal,
            response_completed: responses_protocol.then_some(terminal == TerminalKind::Completed),
            business_committed: value["committed"].as_bool(),
            upstream_http_status: value["upstream_status_code"]
                .as_u64()
                .and_then(|n| u16::try_from(n).ok()),
            // Older producers do not know this fact. Unknown must stay unknown.
            downstream_http_status: value["downstream_http_status"]
                .as_u64()
                .and_then(|n| u16::try_from(n).ok()),
            semantic_status,
        }
    }
    pub fn stream(
        value: &crate::protocols::provider_stream::StreamOutcome,
        upstream_status: u16,
        downstream_status: u16,
    ) -> Self {
        Self {
            terminal: TerminalKind::parse(value.terminal_kind.unwrap_or(if value.success {
                "completed"
            } else {
                "unknown"
            })),
            response_completed: value.response_completed,
            // First output is an observation; completed_usage may still be buffering it.
            business_committed: None,
            upstream_http_status: Some(upstream_status),
            downstream_http_status: Some(downstream_status),
            semantic_status: value.status_code,
        }
    }
}

/// Filled only by the authoritative protocol observer, not body EOF.
#[derive(Clone)]
pub(crate) struct CompletionReceipt(pub tokio::sync::watch::Receiver<Option<bool>>);

impl CompletionReceipt {
    pub fn completed() -> Self {
        Self(tokio::sync::watch::channel(Some(true)).1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn semantic_failure_never_rewrites_the_http_facts() {
        let facts = FinalFacts::native(
            &serde_json::json!({"kind":"semantic_error","upstream_status_code":200,"committed":true}),
            503,
            true,
        );
        assert_eq!(facts.terminal, TerminalKind::SemanticFailure);
        assert_eq!(facts.upstream_http_status, Some(200));
        assert_eq!(facts.downstream_http_status, None);
        assert_eq!(facts.response_completed, Some(false));
    }

    #[test]
    fn opaque_json_completion_remains_unknown_without_a_protocol_receipt() {
        let facts = FinalFacts::buffered(None, 200, 200, false);
        assert_eq!(facts.terminal, TerminalKind::Unknown);
        assert_eq!(facts.response_completed, None);
    }

    #[test]
    fn validated_buffered_responses_terminal_is_recorded_as_complete() {
        let facts = FinalFacts::buffered(
            Some(&serde_json::json!({"kind":"completed"})),
            200,
            200,
            false,
        );
        assert_eq!(facts.terminal, TerminalKind::Completed);
        assert_eq!(facts.response_completed, Some(true));
    }
}
