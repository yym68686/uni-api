//! Observation-only usage. None means the provider did not report a value;
//! reported zero is a measurement. Input includes cached and cache-write tokens.
use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct FactUsage {
    pub input: Option<i64>,
    pub output: Option<i64>,
    pub cache_read: Option<i64>,
    pub cache_write: Option<i64>,
    pub cache_write_1h: Option<i64>,
}

impl FactUsage {
    pub(crate) fn from_usage(usage: Option<&Value>) -> Self {
        let Some(u) = usage.filter(|u| u.is_object()) else {
            return Self::default();
        };
        let read = |paths: &[&str]| {
            paths
                .iter()
                .find_map(|p| u.pointer(p).and_then(Value::as_i64).filter(|n| *n >= 0))
        };
        let cache_read = read(&[
            "/cache_read_input_tokens",
            "/prompt_tokens_details/cached_tokens",
            "/input_tokens_details/cached_tokens",
            "/cachedContentTokenCount",
        ]);
        let write_5m = read(&["/cache_creation/ephemeral_5m_input_tokens"]);
        let cache_write_1h = read(&["/cache_creation/ephemeral_1h_input_tokens"]);
        let split_write = (write_5m.is_some() || cache_write_1h.is_some()).then(|| {
            write_5m
                .unwrap_or(0)
                .saturating_add(cache_write_1h.unwrap_or(0))
        });
        let cache_write = max_optional(
            read(&[
                "/cache_creation_input_tokens",
                "/prompt_tokens_details/cache_write_tokens",
                "/input_tokens_details/cache_write_tokens",
            ]),
            split_write,
        );
        let mut input = read(&["/input_tokens", "/prompt_tokens", "/promptTokenCount"]);
        // Anthropic reports ordinary input separately. OpenAI/Gemini include
        // cached input in their input total, so only the raw Anthropic fields
        // justify addition. Normalized Claude usage retains these raw fields.
        if u.get("cache_read_input_tokens").is_some()
            || u.get("cache_creation_input_tokens").is_some()
            || u.get("cache_creation").is_some()
        {
            input = input.map(|n| {
                n.saturating_add(cache_read.unwrap_or(0))
                    .saturating_add(cache_write.unwrap_or(0))
            });
        }
        let output = read(&[
            "/output_tokens",
            "/completion_tokens",
            "/candidatesTokenCount",
        ])
        .map(|n| n.saturating_add(read(&["/thoughtsTokenCount"]).unwrap_or(0)));
        Self {
            input,
            output,
            cache_read,
            cache_write,
            cache_write_1h,
        }
    }

    /// Stream usage frames are cumulative snapshots, not additive increments.
    pub(crate) fn merge(&mut self, other: Self) {
        self.input = max_optional(self.input, other.input);
        self.output = max_optional(self.output, other.output);
        self.cache_read = max_optional(self.cache_read, other.cache_read);
        self.cache_write = max_optional(self.cache_write, other.cache_write);
        self.cache_write_1h = max_optional(self.cache_write_1h, other.cache_write_1h);
    }
}

fn max_optional(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(v), None) | (None, Some(v)) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn distinguishes_unknown_zero_and_inclusive_cache_accounting() {
        assert_eq!(FactUsage::from_usage(None), FactUsage::default());
        let zero = FactUsage::from_usage(Some(
            &json!({"input_tokens":0,"output_tokens":0,"input_tokens_details":{"cached_tokens":0}}),
        ));
        assert_eq!(
            (zero.input, zero.output, zero.cache_read, zero.cache_write),
            (Some(0), Some(0), Some(0), None)
        );
        let openai = FactUsage::from_usage(Some(
            &json!({"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":60}}),
        ));
        assert_eq!(openai.input, Some(100));
        let claude = FactUsage::from_usage(Some(
            &json!({"input_tokens":5,"output_tokens":13,"cache_read_input_tokens":11,"cache_creation_input_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":5}}),
        ));
        assert_eq!(
            (
                claude.input,
                claude.output,
                claude.cache_read,
                claude.cache_write,
                claude.cache_write_1h
            ),
            (Some(23), Some(13), Some(11), Some(7), Some(5))
        );
    }

    #[test]
    fn cumulative_stream_frames_do_not_double_count_or_erase_cache() {
        let mut value = FactUsage::from_usage(Some(
            &json!({"input_tokens":5,"cache_read_input_tokens":11,"cache_creation_input_tokens":7}),
        ));
        value.merge(FactUsage::from_usage(Some(
            &json!({"input_tokens":0,"cache_read_input_tokens":0,"output_tokens":13}),
        )));
        assert_eq!(
            (
                value.input,
                value.output,
                value.cache_read,
                value.cache_write
            ),
            (Some(23), Some(13), Some(11), Some(7))
        );
        let snapshot = value.clone();
        value.merge(snapshot.clone());
        assert_eq!(value, snapshot);
    }
}
