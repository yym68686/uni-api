//! Read-only upstream billing queries, isolated from routing and model metrics.

use crate::config::snapshot::Provider;
use futures_util::{stream, FutureExt, StreamExt};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Semaphore};
use tokio::time::Instant;
use url::Url;

const TTL: Duration = Duration::from_secs(300);

const MAX_CACHE: usize = 512;

const MAX_KEYS: usize = 8;

const MAX_BODY: usize = 262_144;

type Slot = Arc<Mutex<Option<(Instant, Value)>>>;

struct Balances {
    cache: Mutex<HashMap<String, Slot>>,
    permits: Semaphore,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// Derive only a sibling endpoint on the configured service; never follow a
// redirect or accept arbitrary destination URLs from platform API callers.
fn usage_url(base: &str, start_date: &str, end_date: &str) -> Option<Url> {
    let mut url = Url::parse(base).ok()?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let host = url.host_str()?;
    if matches!(
        host,
        "api.openai.com"
            | "api.anthropic.com"
            | "api.deepseek.com"
            | "generativelanguage.googleapis.com"
            | "aiplatform.googleapis.com"
    ) || host.ends_with("-aiplatform.googleapis.com")
    {
        return None;
    }
    let path = url.path().trim_end_matches('/');
    let prefix = if path.is_empty() {
        ""
    } else if let Some(p) = path
        .strip_suffix("/v1")
        .or_else(|| path.strip_suffix("/v1beta"))
    {
        p
    } else {
        [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/messages",
            "/v1beta/models",
            "/v1/models",
        ]
        .iter()
        .find_map(|suffix| path.strip_suffix(suffix))
        .or_else(|| gemini_usage_prefix(path))?
    };
    let path = format!("{prefix}/v1/usage");
    url.set_path(&path);
    // Ask sub2api for the requested calendar-day range. The response contains
    // provider-reported actual_cost values that are independent of wallet
    // balance changes (including top-ups).
    url.query_pairs_mut()
        .append_pair("start_date", start_date)
        .append_pair("end_date", end_date);
    Some(url)
}

// Native Gemini uses /v1beta instead of /v1, but Sub2API billing is still
// /v1/usage on that same service. Recognize only known model-operation paths;
// do not guess the root of arbitrary URLs or discard a deployment prefix.
fn gemini_usage_prefix(path: &str) -> Option<&str> {
    for marker in ["/v1beta/models/", "/v1/models/"] {
        if let Some((prefix, operation)) = path.rsplit_once(marker) {
            let (model, action) = operation.rsplit_once(':')?;
            if !model.is_empty()
                && !model.contains('/')
                && matches!(
                    action,
                    "generateContent" | "streamGenerateContent" | "countTokens"
                )
            {
                return Some(prefix);
            }
        }
    }
    None
}

fn utc_date(seconds: u64) -> String {
    let shifted = (seconds / 86_400) as i64 + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

pub(crate) fn today_utc() -> String {
    utc_date(now())
}

pub(crate) fn valid_date(value: &str) -> bool {
    value.len() == 10
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 4 | 7) {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        })
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite())
}

fn normalize(value: &Value, model: Option<&str>) -> Option<Value> {
    if !matches!(
        value["mode"].as_str(),
        Some("unrestricted" | "quota_limited")
    ) || !value["isValid"].is_boolean()
    {
        return None;
    }
    let currency = value["unit"].as_str().unwrap_or("USD");
    if currency.len() != 3 || !currency.bytes().all(|c| c.is_ascii_uppercase()) {
        return None;
    }
    let mut windows = Vec::new();
    let (kind, amount, unlimited) = if value["mode"] == "quota_limited" {
        if let Some(remaining) = number(&value["quota"]["remaining"]) {
            ("key_quota", Some(remaining), false)
        } else if let Some(limits) = value["rate_limits"].as_array() {
            for limit in limits.iter().take(8) {
                if let (Some(window), Some(remaining)) =
                    (limit["window"].as_str(), number(&limit["remaining"]))
                {
                    if matches!(window, "5h" | "1d" | "7d") {
                        windows.push(json!({"window":window,"remaining":remaining}));
                    }
                }
            }
            ("key_rate_limits", None, false)
        } else {
            ("key_quota", None, false)
        }
    } else if let Some(balance) = number(&value["balance"]) {
        ("wallet", Some(balance), false)
    } else if value["subscription"].is_object() {
        let remaining = number(&value["remaining"]);
        (
            "subscription",
            remaining.filter(|v| *v >= 0.),
            remaining == Some(-1.),
        )
    } else {
        ("subscription", None, false)
    };
    let mut result = json!({"status":"ok","source":"sub2api","kind":kind,"amount":amount,
        "currency":currency,"unlimited":unlimited,"windows":windows,"key_valid":value["isValid"]});

    // Sub2API exposes actual deductions in usage.total and per-model usage in
    // model_stats. Keep the selected model dimension here so the dashboard can
    // compare its token estimate with the upstream's own charge.
    if currency == "USD" {
        let model_stats = value["model_stats"].as_array();
        let matching = model_stats.map(|stats| {
            stats
                .iter()
                .filter(|row| model.is_none_or(|wanted| row["model"].as_str() == Some(wanted)))
                .collect::<Vec<_>>()
        });
        if let Some(rows) = matching {
            let mut actual = 0.0;
            let mut samples = 0_u64;
            for row in rows {
                if let Some(cost) = number(&row["actual_cost"]) {
                    actual += cost;
                    samples = samples.saturating_add(row["requests"].as_u64().unwrap_or(0));
                }
            }
            result["actual_cost_usd"] = json!(actual);
            result["actual_cost_samples"] = json!(samples);
            result["actual_cost_source"] = json!("sub2api_usage");
        } else if let Some(cost) = number(&value["usage"]["total"]["actual_cost"]) {
            result["actual_cost_usd"] = json!(cost);
            result["actual_cost_samples"] =
                json!(value["usage"]["total"]["requests"].as_u64().unwrap_or(0));
            result["actual_cost_source"] = json!("sub2api_usage");
        }
    }
    Some(result)
}

fn failure(status: &str) -> Value {
    json!({"status":status})
}

impl Balances {
    fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            permits: Semaphore::new(3),
        }
    }

    async fn get(
        &self,
        url: &Url,
        credential: &str,
        proxy: Option<&str>,
        model: Option<&str>,
    ) -> Value {
        let identity =
            serde_json::to_vec(&(url.as_str(), credential, proxy, model)).unwrap_or_default();
        let id = format!("{:x}", Sha256::digest(identity));
        let slot = {
            let mut cache = self.cache.lock().await;
            if !cache.contains_key(&id) && cache.len() >= MAX_CACHE {
                cache.retain(|_, slot| {
                    slot.try_lock().map_or(true, |v| {
                        v.as_ref().is_some_and(|(at, _)| at.elapsed() < TTL)
                    })
                });
                if cache.len() >= MAX_CACHE {
                    return failure("busy");
                }
            }
            cache.entry(id).or_default().clone()
        };
        // Holding this slot coalesces concurrent queries for a shared key.
        let mut cached = slot.lock().await;
        if let Some((at, value)) = cached.as_ref().filter(|(at, _)| at.elapsed() < TTL) {
            let mut value = value.clone();
            value["cached"] = json!(true);
            value["age_seconds"] = json!(at.elapsed().as_secs());
            return value;
        }
        let Ok(Ok(_permit)) =
            tokio::time::timeout(Duration::from_secs(2), self.permits.acquire()).await
        else {
            return failure("busy");
        };
        let mut value = fetch(url, credential, proxy, model).await;
        value["checked_at"] = json!(now());
        value["cached"] = json!(false);
        value["age_seconds"] = json!(0);
        *cached = Some((Instant::now(), value.clone()));
        value
    }
}

async fn fetch(url: &Url, credential: &str, proxy: Option<&str>, model: Option<&str>) -> Value {
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(6));
    if let Some(proxy) = proxy {
        let Ok(proxy) = reqwest::Proxy::all(proxy) else {
            return failure("proxy_error");
        };
        builder = builder.proxy(proxy);
    }
    let Ok(client) = builder.build() else {
        return failure("query_error");
    };
    let response = client
        .get(url.clone())
        .bearer_auth(credential)
        .header("Accept", "application/json")
        .send()
        .await;
    let mut response = match response {
        Ok(response) => response,
        Err(error) => {
            return failure(if error.is_timeout() {
                "timeout"
            } else {
                "network_error"
            })
        }
    };
    let code = response.status().as_u16();
    if code != 200 {
        let mut value = failure(match code {
            401 | 403 => "access_denied",
            404 | 405 => "unsupported",
            429 => "rate_limited",
            300..=399 => "redirect_blocked",
            _ => "upstream_error",
        });
        value["http_status"] = json!(code);
        return value;
    }
    if response
        .content_length()
        .is_some_and(|v| v > MAX_BODY as u64)
    {
        return failure("response_too_large");
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if body.len().saturating_add(chunk.len()) > MAX_BODY {
                    return failure("response_too_large");
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(error) => {
                return failure(if error.is_timeout() {
                    "timeout"
                } else {
                    "network_error"
                })
            }
        }
    }
    serde_json::from_slice(&body)
        .ok()
        .and_then(|v| normalize(&v, model))
        .unwrap_or_else(|| failure("unsupported"))
}

pub(crate) async fn query(
    provider: &Provider,
    proxy: Option<&str>,
    start_date: &str,
    end_date: &str,
    model: Option<&str>,
) -> Value {
    static SERVICE: OnceLock<Balances> = OnceLock::new();
    let service = SERVICE.get_or_init(Balances::new);
    let Some(url) = usage_url(&provider.base_url, start_date, end_date)
        .filter(|_| provider.preferences.get("balance_query") != Some(&Value::Bool(false)))
    else {
        return json!({"provider":provider.name.as_ref(),"status":"unsupported","keys":[]});
    };
    let mut seen = HashSet::new();
    let keys: Vec<_> = provider
        .api_keys
        .iter()
        .filter(|key| seen.insert(key.as_str()))
        .cloned()
        .collect();
    let total = keys.len();
    let mut values: Vec<_> = stream::iter(keys.into_iter().take(MAX_KEYS).enumerate().map(
        |(index, key)| {
            let url = url.clone();
            let proxy = proxy.map(str::to_owned);
            async move {
                let mut value = if key.is_empty() || key.contains(',') || key.contains(['\r', '\n'])
                {
                    failure("unsupported_credential")
                } else {
                    service.get(&url, &key, proxy.as_deref(), model).await
                };
                value["position"] = json!(index + 1);
                value
            }
            .boxed()
        },
    ))
    .buffer_unordered(3)
    .collect()
    .await;
    values.sort_by_key(|v| v["position"].as_u64());
    let actual_values = values
        .iter()
        .filter(|value| value["status"] == "ok" && value.get("actual_cost_usd").is_some());
    let mut actual_cost = 0.0;
    let mut actual_samples = 0_u64;
    let mut actual_available = false;
    for value in actual_values {
        actual_available = true;
        actual_cost += number(&value["actual_cost_usd"]).unwrap_or(0.0);
        actual_samples =
            actual_samples.saturating_add(value["actual_cost_samples"].as_u64().unwrap_or(0));
    }
    let mut result = json!({"provider":provider.name.as_ref(),"status":if total == 0 { "no_key" } else { "complete" },
        "keys":values,"key_count":total,"omitted_keys":total.saturating_sub(MAX_KEYS),"cache_ttl_seconds":TTL.as_secs()});
    if actual_available {
        result["actual_cost_usd"] = json!(actual_cost);
        result["actual_cost_samples"] = json!(actual_samples);
        result["actual_cost_source"] = json!("sub2api_usage");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derives_only_supported_sibling_paths() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(1789385360), "2026-09-14");
        assert_eq!(
            usage_url(
                "https://example.test/prefix/v1/responses",
                "2026-09-01",
                "2026-09-02"
            )
            .unwrap()
            .path(),
            "/prefix/v1/usage"
        );
        assert_eq!(
            usage_url(
                "https://example.test/v1/chat/completions/",
                "2026-09-01",
                "2026-09-02"
            )
            .unwrap()
            .path(),
            "/v1/usage"
        );
        for url in [
            "http://example.test/v1",
            "https://user:secret@example.test/v1",
            "https://example.test/v1?api_key=x",
            "https://api.openai.com/v1",
        ] {
            assert!(usage_url(url, "2026-09-01", "2026-09-02").is_none());
        }
    }
    #[test]
    fn derives_gemini_billing_siblings_without_dropping_deployment_prefixes() {
        for (base, expected) in [
            ("https://site.test/v1beta", "/v1/usage"),
            ("https://site.test/gateway/v1beta/", "/gateway/v1/usage"),
            (
                "https://site.test/gateway/v1beta/models",
                "/gateway/v1/usage",
            ),
            (
                "https://site.test/gateway/v1beta/models/gemini-4-argon:streamGenerateContent",
                "/gateway/v1/usage",
            ),
            (
                "https://site.test/v1/models/gemini-4-argon:generateContent",
                "/v1/usage",
            ),
        ] {
            let url = usage_url(base, "2026-10-01", "2026-10-15").expect(base);
            assert_eq!(url.host_str(), Some("site.test"));
            assert_eq!(url.path(), expected, "{base}");
            assert_eq!(
                url.query(),
                Some("start_date=2026-10-01&end_date=2026-10-15")
            );
        }
        for base in [
            "https://generativelanguage.googleapis.com/v1beta",
            "https://aiplatform.googleapis.com/v1",
            "https://us-central1-aiplatform.googleapis.com/v1",
            "https://site.test/v1beta?key=secret",
            "https://site.test/v1beta#fragment",
            "https://site.test/v1beta/unknown",
            "https://site.test/v1beta/models/a/other",
            "https://site.test/v1beta/models/a:delete",
        ] {
            assert!(
                usage_url(base, "2026-10-01", "2026-10-15").is_none(),
                "{base}"
            );
        }
    }

    #[test]
    fn distinguishes_wallet_quota_subscription_and_missing_data() {
        let wallet = normalize(&json!({"mode":"unrestricted","isValid":true,"balance":0,"unit":"USD","secret":"never return"}), None).unwrap();
        assert_eq!(wallet["amount"], 0.0);
        assert_eq!(wallet["kind"], "wallet");
        assert!(wallet.get("secret").is_none());
        let quota = normalize(&json!({"mode":"quota_limited","isValid":true,"quota":{"remaining":12.5},"balance":900}), None).unwrap();
        assert_eq!(quota["amount"], 12.5);
        assert_eq!(quota["kind"], "key_quota");
        let unlimited = normalize(
            &json!({"mode":"unrestricted","isValid":true,"remaining":-1,"subscription":{}}),
            None,
        )
        .unwrap();
        assert_eq!(unlimited["unlimited"], true);
        assert!(unlimited["amount"].is_null());
        let missing = normalize(&json!({"mode":"unrestricted","isValid":true}), None).unwrap();
        assert!(missing["amount"].is_null());
        assert!(normalize(&json!({"balance":100,"error":"unrelated API"}), None).is_none());
        let limited = normalize(&json!({"mode":"quota_limited","isValid":true,"rate_limits":[{"window":"5h","remaining":3.5}]}), None).unwrap();
        assert_eq!(limited["kind"], "key_rate_limits");
        assert!(limited["amount"].is_null());
        assert_eq!(limited["windows"][0]["remaining"], 3.5);
    }

    #[test]
    fn uses_provider_actual_cost_and_model_filter_instead_of_balance_delta() {
        let payload = json!({
            "mode": "unrestricted",
            "isValid": true,
            "balance": 120.0,
            "unit": "USD",
            "usage": {"total": {"actual_cost": 99.0, "requests": 999}},
            "model_stats": [
                {"model":"model-a","actual_cost":2.5,"requests":4},
                {"model":"model-b","actual_cost":7.0,"requests":3}
            ]
        });
        let selected = normalize(&payload, Some("model-a")).unwrap();
        assert_eq!(selected["actual_cost_usd"], 2.5);
        assert_eq!(selected["actual_cost_samples"], 4);
        let all = normalize(&payload, None).unwrap();
        assert_eq!(all["actual_cost_usd"], 9.5);
        assert_eq!(all["actual_cost_samples"], 7);
    }
    #[tokio::test]
    async fn gemini_sibling_usage_keeps_credentials_and_balance_kinds() {
        use axum::{http::HeaderMap, routing::get, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = Router::new().route(
            "/gateway/v1/usage",
            get(move |headers: HeaderMap| {
                counter.fetch_add(1, Ordering::SeqCst);
                async move {
                    let value = match headers.get("authorization").and_then(|v| v.to_str().ok()) {
                        Some("Bearer wallet-fixture") => json!({"mode":"unrestricted","isValid":true,"balance":57.5}),
                        Some("Bearer quota-fixture") => json!({"mode":"quota_limited","isValid":true,"quota":{"remaining":3},"balance":57.5}),
                        Some("Bearer subscription-fixture") => json!({"mode":"unrestricted","isValid":true,"subscription":{},"remaining":12}),
                        _ => panic!("unexpected billing credential"),
                    };
                    axum::Json(value)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let service = Balances::new();
        let mut url = usage_url(
            "https://site.test/gateway/v1beta",
            "2026-10-01",
            "2026-10-15",
        )
        .unwrap();
        // Redirect the derived URL to this test's isolated HTTP listener only.
        url.set_scheme("http").unwrap();
        url.set_host(Some("127.0.0.1")).unwrap();
        url.set_port(Some(address.port())).unwrap();
        for (credential, kind, amount) in [
            ("wallet-fixture", "wallet", 57.5),
            ("quota-fixture", "key_quota", 3.0),
            ("subscription-fixture", "subscription", 12.0),
        ] {
            let result = service
                .get(&url, credential, None, Some("gemini-4-argon"))
                .await;
            assert_eq!(result["status"], "ok");
            assert_eq!(result["kind"], kind);
            assert_eq!(result["amount"], amount);
            assert!(!result.to_string().contains(credential));
        }
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        task.abort();
    }

    #[tokio::test]
    async fn coalesces_requests_caches_failures_and_blocks_redirects() {
        use axum::{routing::get, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = Router::new()
            .route(
                "/usage",
                get(move || {
                    counter.fetch_add(1, Ordering::SeqCst);
                    async {
                        axum::Json(json!({"mode":"unrestricted","isValid":true,"balance":2.5}))
                    }
                }),
            )
            .route(
                "/redirect",
                get(|| async { axum::response::Redirect::temporary("/usage") }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let service = Balances::new();
        let url = Url::parse(&format!("http://{address}/usage")).unwrap();
        let (a, b) = tokio::join!(
            service.get(&url, "a", None, None),
            service.get(&url, "a", None, None)
        );
        assert_eq!(a["amount"], 2.5);
        assert_eq!(b["amount"], 2.5);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(a["cached"] == true || b["cached"] == true);
        service.get(&url, "different-key", None, None).await;
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let redirect = Url::parse(&format!("http://{address}/redirect")).unwrap();
        assert_eq!(
            service.get(&redirect, "a", None, None).await["status"],
            "redirect_blocked"
        );
        assert_eq!(
            service.get(&redirect, "a", None, None).await["cached"],
            true
        );
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        task.abort();
    }
}
