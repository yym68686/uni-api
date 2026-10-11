//! Responses counters are authoritative over Chat compatibility aliases.
//! A reported zero is a value; absent or invalid usage must not become zero.

use crate::protocols::sse::{parse_sse_frame, SseFrame};
use bytes::Bytes;
use serde_json::{Map, Value};

const ALIASES: [(&str, &str); 4] = [
    ("input_tokens", "prompt_tokens"),
    ("output_tokens", "completion_tokens"),
    ("input_tokens_details", "prompt_tokens_details"),
    ("output_tokens_details", "completion_tokens_details"),
];

pub(crate) fn normalize_payload(payload: &mut Value) -> bool {
    let mut changed = false;
    if let Some(usage) = payload.get_mut("usage") {
        changed |= normalize_aliases(usage);
    }
    if let Some(usage) = payload.pointer_mut("/response/usage") {
        changed |= normalize_aliases(usage);
    }
    changed
}

fn normalize_aliases(usage: &mut Value) -> bool {
    let Some(object) = usage.as_object_mut() else {
        return false;
    };
    let mut changed = false;
    for (native, alias) in ALIASES {
        let Some(value) = object.get(native) else {
            continue;
        };
        // Invalid native values remain visible to the downstream validator.
        let valid = if native.ends_with("_details") {
            value.is_object()
        } else {
            counter(Some(value)).is_some()
        };
        if valid && object.get(alias).is_some_and(|other| other != value) {
            object.insert(alias.into(), value.clone());
            changed = true;
        }
    }
    changed
}

pub(crate) fn to_chat(usage: Option<&Value>) -> Value {
    let Some(source) = usage.and_then(Value::as_object) else {
        return Value::Null;
    };
    let mut mapped = Map::new();
    for (native, alias) in ALIASES {
        if let Some(value) = source.get(native).or_else(|| source.get(alias)) {
            mapped.insert(alias.into(), value.clone());
        }
    }
    for name in ["total_tokens", "oaix_settlement_receipt"] {
        if let Some(value) = source.get(name) {
            mapped.insert(name.into(), value.clone());
        }
    }
    if !mapped.contains_key("total_tokens") {
        if let Some(total) = total_from_counters(&mapped) {
            mapped.insert("total_tokens".into(), total.into());
        }
    }
    Value::Object(mapped)
}

fn counter(value: Option<&Value>) -> Option<i64> {
    value.and_then(Value::as_i64).filter(|value| *value >= 0)
}

fn total_from_counters(usage: &Map<String, Value>) -> Option<i64> {
    counter(usage.get("prompt_tokens"))?.checked_add(counter(usage.get("completion_tokens"))?)
}

/// Preserve every unrelated frame byte-for-byte, including malformed frames:
/// their existing protocol validation and failure handling remain in charge.
pub(crate) fn normalize_frame(frame: SseFrame) -> (SseFrame, bool) {
    let raw = frame.raw();
    if memchr::memmem::find(raw, b"prompt_tokens").is_none()
        && memchr::memmem::find(raw, b"completion_tokens").is_none()
        && memchr::memmem::find(raw, b"\\u").is_none()
    {
        return (frame, false);
    }
    // Usage is normally in a terminal, but some providers report it earlier.
    // Escaped JSON names still get inspected on terminal events.
    if !crate::protocols::responses::processor::terminal_candidate(&frame)
        && memchr::memmem::find(raw, b"usage").is_none()
        && memchr::memmem::find(raw, b"\\u").is_none()
    {
        return (frame, false);
    }
    let Ok(parsed) = parse_sse_frame(&frame) else {
        return (frame, false);
    };
    let Some(data) = parsed.data else {
        return (frame, false);
    };
    let Ok(mut payload) = serde_json::from_str::<Value>(&data) else {
        return (frame, false);
    };
    if !normalize_payload(&mut payload) {
        return (frame, false);
    }
    let Ok(data) = serde_json::to_string(&payload) else {
        return (frame, false);
    };
    // Only replace SSE data lines. Keep event/id/retry/comments and the
    // terminated/unterminated boundary, including CRLF and lone-CR framing.
    let mut wire = Vec::with_capacity(frame.wire.len());
    let mut remaining = frame.raw();
    let mut replaced = false;
    while !remaining.is_empty() {
        let end = memchr::memchr2(b'\r', b'\n', remaining).unwrap_or(remaining.len());
        let line = &remaining[..end];
        let newline_len = if remaining[end..].starts_with(b"\r\n") {
            2
        } else {
            usize::from(end < remaining.len())
        };
        if line == b"data" || line.starts_with(b"data:") {
            if !replaced {
                wire.extend_from_slice(b"data: ");
                wire.extend_from_slice(data.as_bytes());
                wire.extend_from_slice(&remaining[end..end + newline_len]);
                replaced = true;
            }
        } else {
            wire.extend_from_slice(&remaining[..end + newline_len]);
        }
        remaining = &remaining[end + newline_len..];
    }
    let raw_len = wire.len();
    wire.extend_from_slice(&frame.wire[frame.raw_len..]);
    (
        SseFrame {
            wire: Bytes::from(wire),
            raw_len,
            terminated: frame.terminated,
        },
        true,
    )
}

#[cfg(test)]
mod tests;
