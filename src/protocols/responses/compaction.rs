//! Validate explicit compaction attempts without retaining ordinary stream data.
//! Errors are fixed diagnostic categories: never include opaque state or user text.
use serde_json::Value;

#[derive(Default)]
pub(crate) struct CompactionValidator {
    count: usize,
    encrypted: Option<String>,
}

fn compaction(item: &Value) -> bool {
    matches!(
        item["type"].as_str(),
        Some("compaction" | "compaction_summary")
    )
}

pub(crate) fn failure_reason(detail: &str) -> &'static str {
    if detail.starts_with("compaction_invalid_output:") {
        "compaction_invalid_output"
    } else if detail.starts_with("compaction_unsupported:") {
        "compaction_unsupported"
    } else {
        "compaction_stream_failed"
    }
}

// Only explicit capability failures qualify. Auth, quota and arbitrary 400s
// retain their existing policy and diagnostic reason.
pub(crate) fn unsupported(status: u16, body: &str) -> bool {
    if !matches!(status, 400 | 404 | 422 | 501) {
        return false;
    }
    let body = body.to_ascii_lowercase();
    (body.contains("compaction") || body.contains("compaction_trigger"))
        && [
            "not supported",
            "unsupported",
            "unknown type",
            "unrecognized",
        ]
        .iter()
        .any(|marker| body.contains(marker))
}

pub(crate) fn replay(response: Value) -> axum::http::Response<axum::body::Body> {
    let mut created = response.clone();
    created["status"] = "in_progress".into();
    created["output"] = serde_json::json!([]);
    let event = serde_json::json!({"type":"response.created", "response":created});
    let mut wire = format!("event: response.created\ndata: {event}\n\n");
    for (index, item) in response["output"].as_array().unwrap().iter().enumerate() {
        let event = serde_json::json!({"type":"response.output_item.done", "output_index":index, "item":item});
        wire.push_str(&format!(
            "event: response.output_item.done\ndata: {event}\n\n"
        ));
    }
    let event = serde_json::json!({"type":"response.completed", "response":response});
    wire.push_str(&format!("event: response.completed\ndata: {event}\n\n"));
    axum::http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(wire))
        .unwrap()
}

pub(crate) fn validate_response(response: &Value, compact_endpoint: bool) -> Result<(), String> {
    if response.get("error").is_some_and(|v| !v.is_null())
        || !(response["status"] == "completed"
            || (compact_endpoint && response.get("status").is_none()))
    {
        return Err("compaction_stream_failed: response did not complete".into());
    }
    let items = response["output"]
        .as_array()
        .ok_or("compaction_invalid_output: missing output array")?;
    let mut items = items.iter().filter(|item| compaction(item));
    let item = items
        .next()
        .ok_or("compaction_invalid_output: missing compaction item")?;
    if items.next().is_some() {
        return Err("compaction_invalid_output: multiple compaction items".into());
    }
    if item["encrypted_content"]
        .as_str()
        .is_none_or(|v| v.trim().is_empty())
    {
        return Err("compaction_invalid_output: empty or invalid encrypted state".into());
    }
    Ok(())
}

impl CompactionValidator {
    /// Return whether this frame must wait until the attempt is validated.
    pub(crate) fn observe(&mut self, kind: &str, payload: &Value) -> Result<bool, String> {
        match kind {
            "response.failed" | "response.incomplete" | "error" => {
                let error = payload
                    .pointer("/response/error")
                    .or_else(|| payload.get("error"));
                if error.is_some_and(|error| unsupported(400, &error.to_string())) {
                    return Err(
                        "compaction_unsupported: upstream does not support this request".into(),
                    );
                }
                Err("compaction_stream_failed: upstream returned a failure event".into())
            }
            "response.output_item.added" | "response.output_item.done"
                if compaction(&payload["item"]) =>
            {
                if kind == "response.output_item.done" {
                    self.count += 1;
                    if self.count != 1 {
                        return Err(
                            "compaction_invalid_output: multiple compaction done events".into()
                        );
                    }
                    let encrypted = payload["item"]["encrypted_content"]
                        .as_str()
                        .filter(|value| !value.trim().is_empty())
                        .ok_or("compaction_invalid_output: empty or invalid encrypted state")?;
                    self.encrypted = Some(encrypted.to_owned());
                }
                Ok(true)
            }
            "response.completed" => {
                validate_response(&payload["response"], false)?;
                if self.count != 1 {
                    return Err("compaction_invalid_output: missing compaction done event".into());
                }
                let final_item = payload["response"]["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|item| compaction(item))
                    .unwrap();
                if final_item["encrypted_content"].as_str() != self.encrypted.as_deref() {
                    return Err("compaction_invalid_output: inconsistent compaction state".into());
                }
                Ok(true)
            }
            // Codex's shared client caches done items even though the compaction
            // collector ignores ordinary text. Do not cache a losing attempt.
            "response.output_item.done" => Ok(true),
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(state: &str) -> Value {
        json!({"type":"compaction", "encrypted_content":state})
    }
    fn completed(items: Value) -> Value {
        json!({"response":{"status":"completed", "output":items}})
    }

    #[test]
    fn requires_exactly_one_consistent_done_and_terminal() {
        let mut validator = CompactionValidator::default();
        assert!(!validator
            .observe("response.output_text.delta", &json!({"delta":"progress"}))
            .unwrap());
        assert!(validator
            .observe("response.output_item.done", &json!({"item":item("opaque")}))
            .unwrap());
        assert!(validator
            .observe("response.completed", &completed(json!([item("opaque")])))
            .unwrap());
        for (done, output) in [
            (vec![], json!([item("opaque")])),
            (vec![item("opaque")], json!([])),
            (vec![item("opaque")], json!([item("other")])),
            (vec![item("opaque")], json!([item("opaque"), item("other")])),
            (
                vec![item("opaque"), item("opaque")],
                json!([item("opaque")]),
            ),
            (vec![item(" ")], json!([item(" ")])),
        ] {
            let mut validator = CompactionValidator::default();
            let result = done
                .into_iter()
                .try_for_each(|item| {
                    validator
                        .observe("response.output_item.done", &json!({"item":item}))
                        .map(|_| ())
                })
                .and_then(|_| {
                    validator
                        .observe("response.completed", &completed(output))
                        .map(|_| ())
                });
            let error = result.unwrap_err();
            assert!(error.starts_with("compaction_invalid_output:"));
            assert!(!error.contains("opaque"));
        }
    }

    #[test]
    fn native_compact_json_and_capability_failures_are_distinct() {
        let compact = json!({"output":[item("state")]});
        assert!(validate_response(&compact, true).is_ok());
        assert!(validate_response(&compact, false).is_err());
        for status in [401, 403, 429, 500] {
            assert!(!unsupported(status, "compaction unsupported"));
        }
        assert!(unsupported(
            400,
            "unsupported input type: compaction_trigger"
        ));
        assert!(!unsupported(400, "invalid tool arguments"));
        let mut validator = CompactionValidator::default();
        assert!(validator
            .observe("response.failed", &json!({"error":"private"}))
            .unwrap_err()
            .starts_with("compaction_stream_failed:"));
    }
}
