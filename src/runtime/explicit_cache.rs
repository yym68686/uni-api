//! Passive cost routing for one explicit Anthropic text breakpoint. No probes,
//! payload rewrites, accounting changes, or provider-wide penalties.
use crate::config::snapshot::Provider;
use crate::observability::usage::FactUsage;
use crate::providers::types::{AttemptBody, PreparedAttempt, ResponseAdapter};
use hmac::{Hmac, Mac};
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_ENTRIES: usize = 2048;
const MAX_VARIANTS: usize = 16;
const MIN_PREFIX_BYTES: usize = 8192;
const MAX_SUFFIX_BYTES: usize = 2048;
const MAX_BODY_BYTES: usize = 8 << 20;
const MIN_CACHE_TOKENS: i64 = 1024;
const MISS_THRESHOLD: u8 = 2;

type Digest = [u8; 32];

#[derive(Clone)]
pub(crate) struct Guard {
    secret: Option<Arc<Digest>>,
    entries: Arc<Mutex<HashMap<Digest, Entry>>>,
}

impl Default for Guard {
    fn default() -> Self {
        let mut secret = [0; 32];
        let enabled = std::env::var("UNI_API_EXPLICIT_CACHE_GUARD")
            .map(|v| !matches!(v.as_str(), "0" | "off" | "false"))
            .unwrap_or(true);
        Self {
            secret: (enabled && SystemRandom::new().fill(&mut secret).is_ok())
                .then(|| Arc::new(secret)),
            entries: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

pub(crate) struct Candidate {
    scope: Digest,
    variant: Digest,
    prefix: Digest,
    started: Instant,
    ttl: Duration,
    breakpoint: String,
}

struct Entry {
    expires: Instant,
    completed: Instant,
    baseline: i64,
    misses: u8,
    evidence: String,
    variants: VecDeque<(Digest, Instant)>,
}

impl Guard {
    fn hash(&self, parts: &[&[u8]]) -> Option<Digest> {
        let mut mac = Hmac::<Sha256>::new_from_slice(self.secret.as_ref()?.as_ref()).ok()?;
        for part in parts {
            mac.update(&(part.len() as u64).to_le_bytes());
            mac.update(part);
        }
        Some(mac.finalize().into_bytes().into())
    }

    pub(crate) fn candidate(
        &self,
        caller: &str,
        provider: &Provider,
        prepared: &PreparedAttempt,
        endpoint: &str,
    ) -> Option<Candidate> {
        if self.secret.is_none()
            || endpoint != "/v1/messages"
            || prepared.adapter != ResponseAdapter::Passthrough
            || provider.preferences.get("explicit_cache_guard") == Some(&Value::Bool(false))
        {
            return None;
        }
        let AttemptBody::Json(bytes) = &prepared.body else {
            return None;
        };
        if bytes.len() > MAX_BODY_BYTES
            || memchr::memmem::find(bytes, b"\"cache_control\"").is_none()
        {
            return None;
        }
        let root: Value = serde_json::from_slice(bytes).ok()?;
        self.describe(caller, provider, prepared, &root, bytes)
    }

    fn describe(
        &self,
        caller: &str,
        provider: &Provider,
        prepared: &PreparedAttempt,
        root: &Value,
        bytes: &[u8],
    ) -> Option<Candidate> {
        if mark_count(root) != 1 || root.get("cache_control").is_some() {
            return None;
        }
        let mut prefix =
            json!({"tools":root.get("tools"),"system":root.get("system"),"messages":[]});
        let messages = root.get("messages")?.as_array()?;
        let mut suffix = json!({"system":[],"messages":messages});
        let mut mark = None;
        if let Some(system) = root.get("system").and_then(Value::as_array) {
            for (i, block) in system.iter().enumerate() {
                if block.get("cache_control").is_some() {
                    mark = Some((block, format!("system[{i}]")));
                    prefix["system"] = json!(&system[..=i]);
                    suffix["system"] = json!(&system[i + 1..]);
                }
            }
        }
        if mark.is_none() {
            for (i, message) in messages.iter().enumerate() {
                let Some(blocks) = message.get("content").and_then(Value::as_array) else {
                    continue;
                };
                for (j, block) in blocks.iter().enumerate() {
                    if block.get("cache_control").is_some() {
                        mark = Some((block, format!("messages[{i}].content[{j}]")));
                        let mut before = messages[..=i].to_vec();
                        before[i]["content"] = json!(&blocks[..=j]);
                        prefix["messages"] = json!(before);
                        let mut after = messages[i..].to_vec();
                        after[0]["content"] = json!(&blocks[j + 1..]);
                        suffix["messages"] = json!(after);
                    }
                }
            }
        }
        let (block, breakpoint) = mark?;
        if block.get("type")?.as_str()? != "text" || !block.get("text")?.is_string() {
            return None;
        }
        let control = block.get("cache_control")?.as_object()?;
        if control.get("type")?.as_str()? != "ephemeral"
            || control
                .keys()
                .any(|k| !["type", "ttl"].contains(&k.as_str()))
        {
            return None;
        }
        let ttl = match control.get("ttl").and_then(Value::as_str) {
            None if !control.contains_key("ttl") => Duration::from_secs(300),
            Some("5m") => Duration::from_secs(300),
            Some("1h") => Duration::from_secs(3600),
            _ => return None,
        };
        let prefix = serde_json::to_vec(&prefix).ok()?;
        let suffix_len = serde_json::to_vec(&suffix).ok()?.len();
        // Aggregate usage cannot locate arbitrary multiple/internal breakpoints.
        // Require a substantial explicit prefix and a small varying suffix.
        if prefix.len() < MIN_PREFIX_BYTES
            || suffix_len > MAX_SUFFIX_BYTES
            || suffix_len > prefix.len() / 4
        {
            return None;
        }
        let mut parameters = root.as_object()?.clone();
        for field in ["tools", "system", "messages"] {
            parameters.remove(field);
        }
        let parameters = serde_json::to_vec(&parameters).ok()?;
        let preferences = serde_json::to_vec(provider.preferences.as_ref()).ok()?;
        let mut headers = Vec::new();
        for name in ["anthropic-version", "anthropic-beta"] {
            for value in prepared.headers.get_all(name) {
                headers.extend_from_slice(value.as_bytes());
                headers.push(0);
            }
        }
        let credential =
            crate::observability::billing::request_key(&prepared.headers, &prepared.url);
        Some(Candidate {
            scope: self.hash(&[
                caller.as_bytes(),
                provider.name.as_bytes(),
                prepared.url.as_bytes(),
                credential.as_bytes(),
                prepared.original_model.as_bytes(),
                &preferences,
                &parameters,
                &headers,
                &prefix,
            ])?,
            variant: self.hash(&[bytes])?,
            prefix: self.hash(&[&prefix])?,
            started: Instant::now(),
            ttl,
            breakpoint,
        })
    }

    pub(crate) fn avoidance(&self, candidate: &Candidate) -> Option<Value> {
        let entries = self.entries.try_lock().ok()?;
        let entry = entries.get(&candidate.scope)?;
        let now = Instant::now();
        if now >= entry.expires
            || candidate.started < entry.completed
            || entry.misses < MISS_THRESHOLD
            || entry
                .variants
                .iter()
                .any(|(v, expires)| v == &candidate.variant && *expires > now)
        {
            return None;
        }
        Some(detail(candidate, entry, "avoid", None))
    }

    pub(crate) fn observe(
        &self,
        candidate: Option<&Candidate>,
        usage: &FactUsage,
        success: bool,
        request_id: &str,
    ) -> Option<Value> {
        let candidate = candidate?;
        if !success {
            return None;
        }
        let (read, write) = (usage.cache_read?, usage.cache_write?);
        let total = read.checked_add(write)?;
        if read < 0 || write < 0 || total < MIN_CACHE_TOKENS {
            return None;
        }
        let now = Instant::now();
        let expires = candidate.started + candidate.ttl;
        if now >= expires {
            return None;
        }
        let mut entries = self.entries.try_lock().ok()?;
        if entries.len() >= MAX_ENTRIES {
            entries.retain(|_, e| e.expires > now);
            if entries.len() >= MAX_ENTRIES && !entries.contains_key(&candidate.scope) {
                return None;
            }
        }
        let entry = entries.entry(candidate.scope).or_insert_with(|| Entry {
            expires,
            completed: now,
            baseline: total,
            misses: 0,
            evidence: String::new(),
            variants: VecDeque::new(),
        });
        let expired = entry.expires <= candidate.started;
        if expired {
            *entry = Entry {
                expires,
                completed: now,
                baseline: total,
                misses: 0,
                evidence: String::new(),
                variants: VecDeque::new(),
            };
        }
        // Requests overlapping an earlier completion cannot prove reuse failure.
        let sequential =
            expired || entry.variants.is_empty() || candidate.started >= entry.completed;
        let known = entry
            .variants
            .iter()
            .any(|(v, until)| v == &candidate.variant && *until > candidate.started);
        entry.baseline = entry.baseline.min(total);
        let miss = sequential
            && !known
            && !entry.variants.is_empty()
            && read < entry.baseline / 2
            && write >= entry.baseline / 2;
        let reused = sequential
            && !known
            && !entry.variants.is_empty()
            && read >= entry.baseline - entry.baseline / 5
            && write <= entry.baseline / 5;
        let action = if miss {
            entry.misses = entry.misses.saturating_add(1);
            entry.evidence = request_id.chars().take(128).collect();
            "rewrite_observed"
        } else if reused {
            entry.misses = 0;
            entry.evidence.clear();
            entry.expires = expires;
            "prefix_reused"
        } else {
            "warmup_or_repeat"
        };
        if !miss && known && read >= entry.baseline - entry.baseline / 5 {
            entry.expires = expires;
        }
        entry.completed = entry.completed.max(now);
        entry
            .variants
            .retain(|(v, until)| v != &candidate.variant && *until > now);
        entry.variants.push_back((candidate.variant, expires));
        while entry.variants.len() > MAX_VARIANTS {
            entry.variants.pop_front();
        }
        Some(detail(candidate, entry, action, Some((read, write))))
    }
}

fn detail(candidate: &Candidate, entry: &Entry, action: &str, usage: Option<(i64, i64)>) -> Value {
    json!({"cache_guard_action":action,"cache_prefix_hmac":hex::encode(candidate.prefix),
        "cache_breakpoint":candidate.breakpoint,"cache_ttl_seconds":candidate.ttl.as_secs(),
        "cache_rewrite_count":entry.misses,"cache_baseline_tokens":entry.baseline,
        "cache_evidence_request_id":entry.evidence,"cache_avoid_remaining_seconds":entry.expires.saturating_duration_since(Instant::now()).as_secs(),
        "cache_observed_read_tokens":usage.map(|v|v.0),"cache_observed_write_tokens":usage.map(|v|v.1)})
}

fn mark_count(v: &Value) -> usize {
    match v {
        Value::Object(o) => o
            .iter()
            .map(|(k, v)| usize::from(k == "cache_control") + mark_count(v))
            .sum(),
        Value::Array(a) => a.iter().map(mark_count).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::snapshot::{runtime_provider, RawProvider};
    use crate::providers::types::DownstreamProtocol;
    use axum::http::{HeaderMap, Method};
    use std::sync::atomic::AtomicUsize;

    fn fixture(question: &str) -> (Arc<Provider>, PreparedAttempt) {
        let provider = runtime_provider(serde_json::from_value::<RawProvider>(json!({
            "name":"supplier","base_url":"https://supplier.invalid/v1/messages","engine":"claude",
            "api":["private-upstream-key"],"models":{"claude":"claude"}
        })).unwrap(), Arc::new(AtomicUsize::new(0)));
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "private-upstream-key".parse().unwrap());
        let p = PreparedAttempt { dispatch:None, method:Method::POST,
            url:"https://supplier.invalid/v1/messages".into(),headers,
            body:AttemptBody::Json(serde_json::to_vec(&json!({"model":"claude","stream":true,"max_tokens":24,
                "system":[{"type":"text","text":"reference ".repeat(1000),"cache_control":{"type":"ephemeral","ttl":"5m"}}],
                "messages":[{"role":"user","content":question}]})).unwrap()),
            adapter:ResponseAdapter::Passthrough,downstream_stream:true,upstream_stream:true,
            request_model:"claude".into(),original_model:"claude".into(),request_type:None,wire_model:None,
            downstream_protocol:DownstreamProtocol::Native,chat_stream_include_usage:false,
            provider_key:"private-upstream-key".into(),estimated_video_tokens:None };
        (provider, p)
    }
    fn candidate(guard: &Guard, q: &str) -> Candidate {
        let (provider, p) = fixture(q);
        guard
            .candidate("caller-one", &provider, &p, "/v1/messages")
            .unwrap()
    }
    fn usage(read: i64, write: i64) -> FactUsage {
        FactUsage {
            cache_read: Some(read),
            cache_write: Some(write),
            ..Default::default()
        }
    }
    fn learn(guard: &Guard) {
        for q in ["A", "B", "C"] {
            guard
                .observe(Some(&candidate(guard, q)), &usage(650, 4100), true, q)
                .unwrap();
        }
    }
    #[test]
    fn avoids_only_new_suffixes_for_the_exact_failed_prefix_and_scope() {
        let g = Guard::default();
        let a = candidate(&g, "A");
        g.observe(Some(&a), &usage(650, 4100), true, "warm");
        g.observe(Some(&candidate(&g, "A")), &usage(4750, 0), true, "repeat");
        g.observe(Some(&candidate(&g, "B")), &usage(650, 4100), true, "miss-b");
        assert!(g.avoidance(&candidate(&g, "C")).is_none());
        g.observe(Some(&candidate(&g, "C")), &usage(650, 4100), true, "miss-c");
        let detail = g.avoidance(&candidate(&g, "D")).unwrap();
        assert_eq!(detail["cache_rewrite_count"], 2);
        assert_eq!(detail["cache_evidence_request_id"], "miss-c");
        for q in ["A", "B", "C"] {
            assert!(g.avoidance(&candidate(&g, q)).is_none());
        }
        let (p, mut wire) = fixture("D");
        let AttemptBody::Json(bytes) = &mut wire.body else {
            unreachable!()
        };
        let mut body: Value = serde_json::from_slice(bytes).unwrap();
        body["system"][0]["text"] = json!("other reference ".repeat(1000));
        *bytes = serde_json::to_vec(&body).unwrap();
        assert!(g
            .avoidance(
                &g.candidate("caller-one", &p, &wire, "/v1/messages")
                    .unwrap()
            )
            .is_none());
        assert!(g
            .avoidance(
                &g.candidate("caller-two", &p, &fixture("D").1, "/v1/messages")
                    .unwrap()
            )
            .is_none());
        wire = fixture("D").1;
        wire.headers
            .insert("x-api-key", "different-key".parse().unwrap());
        assert!(g
            .avoidance(
                &g.candidate("caller-one", &p, &wire, "/v1/messages")
                    .unwrap()
            )
            .is_none());
        wire = fixture("D").1;
        wire.original_model = "different-model".into();
        assert!(g
            .avoidance(
                &g.candidate("caller-one", &p, &wire, "/v1/messages")
                    .unwrap()
            )
            .is_none());
        wire = fixture("D").1;
        wire.headers
            .insert("anthropic-beta", "different-beta".parse().unwrap());
        assert!(g
            .avoidance(
                &g.candidate("caller-one", &p, &wire, "/v1/messages")
                    .unwrap()
            )
            .is_none());
        let raw = detail.to_string();
        assert!(
            !raw.contains("reference")
                && !raw.contains("private-upstream-key")
                && !raw.contains("caller-one")
        );
    }
    #[test]
    fn leaves_unmarked_multiple_short_and_large_suffix_requests_unclassified() {
        let g = Guard::default();
        for kind in [
            "unmarked",
            "multiple",
            "short",
            "large_suffix",
            "automatic",
            "invalid_ttl",
        ] {
            let (p, mut w) = fixture("A");
            let AttemptBody::Json(bytes) = &mut w.body else {
                unreachable!()
            };
            let mut body: Value = serde_json::from_slice(bytes).unwrap();
            match kind {
                "unmarked" => {
                    body["system"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("cache_control");
                }
                "multiple" => {
                    body["messages"][0]["content"] =
                        json!([{"type":"text","text":"A","cache_control":{"type":"ephemeral"}}]);
                }
                "short" => body["system"][0]["text"] = json!("short"),
                "large_suffix" => body["messages"][0]["content"] = json!("large".repeat(1000)),
                "automatic" => {
                    body["system"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("cache_control");
                    body["cache_control"] = json!({"type":"ephemeral"});
                }
                _ => body["system"][0]["cache_control"]["ttl"] = json!("invalid"),
            }
            *bytes = serde_json::to_vec(&body).unwrap();
            assert!(
                g.candidate("caller", &p, &w, "/v1/messages").is_none(),
                "{kind}"
            );
        }
    }
    #[test]
    fn ignores_overlap_failures_unknown_usage_and_repeat_write_bills() {
        let g = Guard::default();
        let a = candidate(&g, "A");
        let b = candidate(&g, "B");
        g.observe(Some(&a), &usage(650, 4100), true, "a");
        g.observe(Some(&b), &usage(650, 4100), true, "overlap");
        for _ in 0..3 {
            g.observe(Some(&candidate(&g, "A")), &usage(650, 4100), true, "repeat");
        }
        g.observe(
            Some(&candidate(&g, "C")),
            &usage(650, 4100),
            false,
            "failed",
        );
        g.observe(
            Some(&candidate(&g, "D")),
            &FactUsage::default(),
            true,
            "unknown",
        );
        assert!(g.avoidance(&candidate(&g, "E")).is_none());
    }
    #[test]
    fn expiry_and_new_suffix_reuse_clear_avoidance_but_repeat_hits_do_not() {
        let g = Guard::default();
        learn(&g);
        g.observe(
            Some(&candidate(&g, "A")),
            &usage(4750, 0),
            true,
            "repeat-hit",
        );
        assert!(g.avoidance(&candidate(&g, "D")).is_some());
        g.observe(Some(&candidate(&g, "D")), &usage(4750, 0), true, "new-hit");
        assert!(g.avoidance(&candidate(&g, "E")).is_none());
        let expired = Guard::default();
        learn(&expired);
        for e in expired.entries.lock().unwrap().values_mut() {
            e.expires = Instant::now() - Duration::from_secs(1);
        }
        assert!(expired.avoidance(&candidate(&expired, "D")).is_none());
    }
}
