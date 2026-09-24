use crate::transport::spool::SpoolObservation;
use crate::transport::spool::StoredBody;
use axum::http::{HeaderMap, Method};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResponseAdapter {
    Passthrough,
    Search,
    ResponsesToChat,
    GeminiToChat,
    ClaudeToChat,
    CohereToChat,
    CloudflareToChat,
    AwsToChat,
    LingjingVideo,
    CallxyqVideo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DownstreamProtocol {
    Native,
    ResponsesCompat,
}

pub(crate) struct PreparedAttempt {
    pub(crate) dispatch: Option<crate::observability::timing::AttemptDispatch>,
    pub(crate) method: Method,
    pub(crate) url: String,
    pub(crate) headers: HeaderMap,
    pub(crate) body: AttemptBody,
    pub(crate) adapter: ResponseAdapter,
    pub(crate) downstream_stream: bool,
    pub(crate) upstream_stream: bool,
    pub(crate) request_model: String,
    pub(crate) original_model: String,
    pub(crate) wire_model: Option<String>,
    pub(crate) downstream_protocol: DownstreamProtocol,
    pub(crate) chat_stream_include_usage: bool,
    pub(crate) provider_key: String,
    pub(crate) estimated_video_tokens: Option<i64>,
}

pub(crate) enum AttemptBody {
    Json(Vec<u8>),
    Replay(StoredBody, SpoolObservation),
    MultipartRewrite {
        storage: StoredBody,
        observation: SpoolObservation,
        source_content_type: String,
        boundary: String,
        model: String,
    },
    DashscopeTranscription {
        storage: StoredBody,
        observation: SpoolObservation,
        source_content_type: String,
        model: String,
        provider_key: String,
    },
    Empty,
}

pub(crate) struct PreparedInput {
    pub(crate) payload: Option<Value>,
    pub(crate) replay: Option<(StoredBody, SpoolObservation)>,
    pub(crate) observation: SpoolObservation,
    pub(crate) default_model: String,
    pub(crate) content_type: String,
}
