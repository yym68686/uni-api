use crate::observability::stream::StreamStats;
use crate::protocols::responses::events::encode_event;
use crate::protocols::responses::events::extract_usage;
use crate::protocols::responses::events::has_real_output;
use crate::protocols::responses::events::is_canonical_keepalive;
use crate::protocols::responses::events::semantic_failure;
use crate::protocols::responses::events::validate_terminal;
use crate::protocols::responses::events::Terminal;
use crate::protocols::responses::item_ids::event_item_id_needs_normalization;
use crate::protocols::responses::item_ids::ResponsesItemIdNormalizer;
use crate::protocols::sse::parse_sse_frame;
use crate::protocols::sse::SseFrame;
use bytes::Bytes;
use memchr::memmem;
use serde_json::Value;

#[derive(Debug)]
pub(crate) struct ProcessedEvent {
    pub(crate) wire: Option<Bytes>,
    pub(crate) event_type: Option<String>,
    pub(crate) commits: bool,
    pub(crate) canonical_keepalive: bool,
    pub(crate) terminal: Option<Terminal>,
}

pub(crate) struct ResponsesProcessor {
    pub(crate) commit_policy: String,
    pub(crate) normalizer: Option<ResponsesItemIdNormalizer>,
}

impl ResponsesProcessor {
    pub(crate) fn new(commit_policy: String, normalize_ids: bool) -> Self {
        let commit_policy = match commit_policy.trim().to_ascii_lowercase().as_str() {
            "completed_usage" => "completed_usage".to_owned(),
            _ => "real_output".to_owned(),
        };
        Self {
            commit_policy,
            normalizer: normalize_ids.then(ResponsesItemIdNormalizer::default),
        }
    }

    pub(crate) fn process(
        &mut self,
        frame: SseFrame,
        stats: &mut StreamStats,
    ) -> Result<ProcessedEvent, String> {
        stats.event_count = stats.event_count.saturating_add(1);
        let parsed = parse_sse_frame(&frame)?;
        if parsed.comment_only {
            if parsed.raw.starts_with(": oaix-terminal-flush-v1 ") {
                return Ok(ProcessedEvent {
                    wire: None,
                    event_type: None,
                    commits: false,
                    canonical_keepalive: false,
                    terminal: None,
                });
            }
            return Ok(ProcessedEvent {
                wire: Some(frame.canonical_wire()),
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        }
        let Some(data) = parsed.data else {
            return Ok(ProcessedEvent {
                wire: None,
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        };
        if data.trim() == "[DONE]" {
            return Err(
                "Responses upstream emitted [DONE] without a terminal response event".into(),
            );
        }
        let mut payload: Value = serde_json::from_str(&data)
            .map_err(|error| format!("Responses upstream event JSON is invalid: {error}"))?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "Responses upstream event payload is missing a string type".to_owned())?
            .to_owned();
        if event_type.len() > 256 || event_type.contains(['\r', '\n']) {
            return Err("Responses upstream event type is invalid".into());
        }
        if let Some(declared) = parsed.declared_event.as_deref() {
            if declared != event_type {
                return Err(format!(
                    "Responses event field {declared:?} does not match payload type {event_type:?}"
                ));
            }
        }
        validate_terminal(&event_type, &payload)?;

        let normalized = if let Some(normalizer) = self.normalizer.as_mut() {
            normalizer.normalize(&mut payload)?
        } else {
            false
        };
        let wire = if normalized {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            encode_event(&event_type, &payload)?
        } else if parsed.declared_event.is_none() {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            Bytes::from(format!("event: {event_type}\n{}\n\n", parsed.raw))
        } else {
            frame.canonical_wire()
        };
        if event_type.ends_with(".delta") {
            stats.delta_events = stats.delta_events.saturating_add(1);
        }
        if let Some(usage) = extract_usage(&payload) {
            stats.usage = Some(usage.clone());
        }

        let terminal = if semantic_failure(&event_type, &payload) {
            Some(Terminal::SemanticFailure {
                event_type: event_type.clone(),
                payload: payload.clone(),
            })
        } else if event_type == "response.completed" {
            Some(Terminal::Completed)
        } else if event_type == "response.incomplete" {
            Some(Terminal::Incomplete)
        } else {
            None
        };
        let canonical_keepalive = event_type == "keepalive" && is_canonical_keepalive(&payload);
        stats.observe_semantic_output(&event_type, &payload);
        let commits = terminal.is_some()
            || (!matches!(
                event_type.as_str(),
                "response.created" | "response.in_progress" | "response.queued" | "keepalive"
            ) && self.commit_policy != "completed_usage"
                && has_real_output(&event_type, &payload));
        Ok(ProcessedEvent {
            wire: Some(wire),
            event_type: Some(event_type),
            commits,
            canonical_keepalive,
            terminal,
        })
    }
}

pub(crate) fn declared_event_bytes(raw: &[u8]) -> Option<&[u8]> {
    for line in raw.split(|byte| matches!(*byte, b'\r' | b'\n')) {
        let Some(mut value) = line.strip_prefix(b"event:") else {
            continue;
        };
        if let Some(stripped) = value.strip_prefix(b" ") {
            value = stripped;
        }
        return Some(value);
    }
    None
}

pub(crate) fn frame_has_data(raw: &[u8]) -> bool {
    raw.split(|byte| matches!(*byte, b'\r' | b'\n'))
        .any(|line| line == b"data" || line.starts_with(b"data:"))
}

pub(crate) fn frame_is_comment_only(raw: &[u8]) -> bool {
    let mut saw_comment = false;
    for line in raw.split(|byte| matches!(*byte, b'\r' | b'\n')) {
        if line.is_empty() {
            continue;
        }
        if line.starts_with(b":") {
            saw_comment = true;
        } else {
            return false;
        }
    }
    saw_comment
}

pub(crate) fn is_terminal_event_type(event_type: &[u8]) -> bool {
    matches!(
        event_type,
        b"error" | b"response.completed" | b"response.failed" | b"response.incomplete"
    )
}

pub(crate) fn terminal_candidate(frame: &SseFrame) -> bool {
    // A data-only event's type is in JSON and may use arbitrary whitespace.
    if declared_event_bytes(frame.raw()).is_none() && frame_has_data(frame.raw()) {
        if let Ok(parsed) = parse_sse_frame(frame) {
            if let Some(data) = parsed.data {
                if let Ok(payload) = serde_json::from_str::<Value>(&data) {
                    if payload
                        .get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|event| is_terminal_event_type(event.as_bytes()))
                    {
                        return true;
                    }
                }
            }
        }
    }
    if declared_event_bytes(frame.raw()).is_some_and(is_terminal_event_type) {
        return true;
    }
    [
        b"response.completed".as_slice(),
        b"response.failed".as_slice(),
        b"response.incomplete".as_slice(),
        b"\"type\":\"error\"".as_slice(),
    ]
    .iter()
    .any(|needle| memmem::find(frame.raw(), needle).is_some())
}

pub(crate) fn selective_rewrite_candidate(frame: &SseFrame) -> bool {
    let raw = frame.raw();
    let declared_event = declared_event_bytes(raw);
    terminal_candidate(frame)
        || declared_event.is_none()
        || !frame_has_data(raw)
        || declared_event.is_some_and(|event_type| {
            matches!(
                event_type,
                b"response.output_item.added" | b"response.output_item.done"
            ) || json_item_id_fields(raw)
                .any(|item_id| event_item_id_needs_normalization(event_type, item_id))
        })
}

pub(crate) fn json_item_id_fields(raw: &[u8]) -> impl Iterator<Item = &[u8]> {
    const ITEM_ID_KEY: &[u8] = b"\"item_id\"";
    memmem::find_iter(raw, ITEM_ID_KEY).filter_map(|key_start| {
        let after_field = raw.get(key_start + ITEM_ID_KEY.len()..)?;
        let colon = after_field.iter().position(|byte| *byte == b':')?;
        let after_colon = after_field.get(colon + 1..)?;
        let quote = after_colon
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())?;
        let value = after_colon.get(quote..)?.strip_prefix(b"\"")?;
        let end = value.iter().position(|byte| *byte == b'\"')?;
        value.get(..end)
    })
}

pub(crate) fn observe_light_frame(stats: &mut StreamStats, frame: &SseFrame) {
    stats.event_count = stats.event_count.saturating_add(1);
    if declared_event_bytes(frame.raw()).is_some_and(|event_type| event_type.ends_with(b".delta")) {
        stats.delta_events = stats.delta_events.saturating_add(1);
    }
}

pub(crate) fn inspect_terminal_frame(
    frame: &SseFrame,
    stats: &mut StreamStats,
) -> Result<Option<Terminal>, String> {
    observe_light_frame(stats, frame);
    if stats.first_output_ms.is_none() || stats.first_text_ms.is_none() {
        // Observation is best-effort and cannot reject or rewrite raw traffic.
        if let Ok(parsed) = parse_sse_frame(frame) {
            if let Some(data) = parsed.data {
                if let Ok(payload) = serde_json::from_str::<Value>(&data) {
                    let kind = payload
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    stats.observe_semantic_output(kind, &payload);
                }
            }
        }
    }
    if !terminal_candidate(frame) {
        return Ok(None);
    }
    let parsed = parse_sse_frame(frame)?;
    let Some(data) = parsed.data else {
        return Ok(None);
    };
    if data.trim() == "[DONE]" {
        return Err("Responses upstream emitted [DONE] without a terminal response event".into());
    }
    let payload: Value = serde_json::from_str(&data)
        .map_err(|error| format!("Responses upstream terminal JSON is invalid: {error}"))?;
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "Responses upstream terminal payload is missing a string type".to_owned())?;
    if !is_terminal_event_type(event_type.as_bytes()) {
        return Ok(None);
    }
    if let Some(declared) = parsed.declared_event.as_deref() {
        if declared != event_type {
            return Err(format!(
                "Responses event field {declared:?} does not match payload type {event_type:?}"
            ));
        }
    }
    validate_terminal(event_type, &payload)?;
    if let Some(usage) = extract_usage(&payload) {
        stats.usage = Some(usage.clone());
    }
    if semantic_failure(event_type, &payload) {
        return Ok(Some(Terminal::SemanticFailure {
            event_type: event_type.to_owned(),
            payload,
        }));
    }
    Ok(match event_type {
        "response.completed" => Some(Terminal::Completed),
        "response.incomplete" => Some(Terminal::Incomplete),
        _ => None,
    })
}
