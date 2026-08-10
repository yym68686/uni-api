use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Response, StatusCode};
use axum::response::IntoResponse;
use tokio::sync::Mutex;

use crate::responses;

const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";
const INTERNAL_PREFIX: &str = "/_internal/rust-responses";

#[derive(Clone)]
pub struct AppState {
    pub backend_origin: Arc<str>,
    pub control_token: Arc<str>,
    pub backend_client: reqwest::Client,
    upstream_clients: Arc<Mutex<HashMap<ClientKey, reqwest::Client>>>,
    resource_guard: ResourceGuard,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ClientKey {
    proxy: Option<String>,
    http1_only: bool,
}

impl AppState {
    pub fn new(backend_origin: String, control_token: String) -> Result<Self, reqwest::Error> {
        let backend_client = reqwest::Client::builder()
            .http1_only()
            .pool_max_idle_per_host(256)
            .build()?;
        Ok(Self {
            backend_origin: backend_origin.into(),
            control_token: control_token.into(),
            backend_client,
            upstream_clients: Arc::new(Mutex::new(HashMap::new())),
            resource_guard: ResourceGuard::new(),
        })
    }

    pub async fn upstream_client(
        &self,
        proxy: Option<&str>,
        http1_only: bool,
    ) -> Result<reqwest::Client, String> {
        let key = ClientKey {
            proxy: proxy.map(str::to_owned),
            http1_only,
        };
        if let Some(client) = self.upstream_clients.lock().await.get(&key).cloned() {
            return Ok(client);
        }

        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(256)
            .tcp_keepalive(std::time::Duration::from_secs(30));
        if http1_only {
            builder = builder.http1_only();
        }
        if let Some(proxy_url) = proxy.filter(|value| !value.trim().is_empty()) {
            let configured = reqwest::Proxy::all(proxy_url)
                .map_err(|error| format!("invalid upstream proxy: {error}"))?;
            builder = builder.proxy(configured);
        }
        let client = builder
            .build()
            .map_err(|error| format!("upstream client build failed: {error}"))?;
        self.upstream_clients
            .lock()
            .await
            .insert(key, client.clone());
        Ok(client)
    }

    pub fn internal_url(&self, path: &str) -> String {
        format!("{}{}", self.backend_origin, path)
    }

    fn resource_rejection(&self) -> Option<&'static str> {
        self.resource_guard.rejection()
    }
}

pub async fn handler(State(state): State<AppState>, request: Request) -> Response<Body> {
    let path = request.uri().path().to_owned();
    if path.starts_with(INTERNAL_PREFIX) {
        return json_error(StatusCode::NOT_FOUND, "Not found");
    }
    if path != "/healthz" {
        if let Some(reason) = state.resource_rejection() {
            let mut response = json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Local memory, file-descriptor, or ephemeral-port headroom is exhausted",
            );
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
            response.headers_mut().insert(
                "x-uni-api-status-origin",
                HeaderValue::from_static("rust_local_admission"),
            );
            if let Ok(value) = HeaderValue::from_str(reason) {
                response
                    .headers_mut()
                    .insert("x-uni-api-admission-reason", value);
            }
            return response;
        }
    }

    let has_idempotency_key = request.headers().contains_key("idempotency-key");
    let use_rust_responses =
        request.method() == Method::POST && path == "/v1/responses" && !has_idempotency_key;
    match proxy_to_backend(&state, request, use_rust_responses).await {
        Ok((response, session_id)) => {
            if let Some(session_id) = session_id {
                responses::serve_session(state, session_id, response).await
            } else {
                relay_response(response)
            }
        }
        Err(error) => json_error(
            StatusCode::BAD_GATEWAY,
            &format!("Python control plane unavailable: {error}"),
        ),
    }
}

async fn proxy_to_backend(
    state: &AppState,
    request: Request,
    rust_responses: bool,
) -> Result<(reqwest::Response, Option<String>), reqwest::Error> {
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", state.backend_origin, path_and_query);
    let mut headers = filtered_request_headers(&parts.headers);
    if rust_responses {
        if let Ok(value) = HeaderValue::from_str(&state.control_token) {
            headers.insert(HeaderName::from_static(CONTROL_HEADER), value);
        }
    }
    let body_stream = body.into_data_stream();
    let response = state
        .backend_client
        .request(parts.method, url)
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body_stream))
        .send()
        .await?;
    let session_id = response
        .headers()
        .get("x-uni-api-rust-responses-session")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    Ok((response, session_id))
}

pub fn relay_response(response: reqwest::Response) -> Response<Body> {
    let status = response.status();
    let headers = filtered_response_headers(response.headers());
    let stream = response.bytes_stream();
    let mut output = Response::new(Body::from_stream(stream));
    *output.status_mut() = status;
    *output.headers_mut() = headers;
    output
}

pub fn filtered_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str())
            || name.as_str().eq_ignore_ascii_case("host")
            || name.as_str().eq_ignore_ascii_case(CONTROL_HEADER)
            || name
                .as_str()
                .eq_ignore_ascii_case("x-uni-api-rust-responses-session")
        {
            continue;
        }
        filtered.append(name.clone(), value.clone());
    }
    filtered
}

pub fn filtered_response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str())
            || name.as_str().eq_ignore_ascii_case("content-length")
            || name.as_str().eq_ignore_ascii_case("content-encoding")
            || name
                .as_str()
                .eq_ignore_ascii_case("x-uni-api-rust-responses-session")
        {
            continue;
        }
        filtered.append(name.clone(), value.clone());
    }
    filtered
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

pub fn json_error(status: StatusCode, message: &str) -> Response<Body> {
    (
        status,
        [("content-type", "application/json")],
        serde_json::json!({"error": {"message": message}}).to_string(),
    )
        .into_response()
}

#[derive(Clone)]
struct ResourceGuard {
    cached: Arc<std::sync::Mutex<CachedResourceSample>>,
    sample_interval: Duration,
}

struct CachedResourceSample {
    sampled_at: Instant,
    rejection: Option<&'static str>,
}

impl ResourceGuard {
    fn new() -> Self {
        Self {
            cached: Arc::new(std::sync::Mutex::new(CachedResourceSample {
                sampled_at: Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .unwrap_or_else(Instant::now),
                rejection: None,
            })),
            sample_interval: Duration::from_millis(
                std::env::var("RUST_RESOURCE_SAMPLE_INTERVAL_MS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .filter(|value: &u64| *value > 0)
                    .unwrap_or(250),
            ),
        }
    }

    fn rejection(&self) -> Option<&'static str> {
        let mut cached = match self.cached.lock() {
            Ok(cached) => cached,
            Err(_) => return None,
        };
        if cached.sampled_at.elapsed() < self.sample_interval {
            return cached.rejection;
        }
        cached.sampled_at = Instant::now();
        cached.rejection = sample_resource_rejection();
        cached.rejection
    }
}

fn sample_resource_rejection() -> Option<&'static str> {
    if let Some((current, limit)) = cgroup_memory() {
        let configured_guard = std::env::var("RUST_FRONTEND_MEMORY_GUARD_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(64 * 1024 * 1024);
        let guard = configured_guard.max(limit / 20);
        if current.saturating_add(guard) >= limit {
            return Some("memory_headroom");
        }
    }
    if let Some((open, limit)) = file_descriptor_usage() {
        let reserve = 64_u64.max(limit / 20);
        if open.saturating_add(reserve) >= limit {
            return Some("file_descriptor_headroom");
        }
    }
    if let Some((used, total)) = ephemeral_port_usage() {
        let reserve = 256_u64.max(total / 20);
        if used.saturating_add(reserve) >= total {
            return Some("ephemeral_port_headroom");
        }
    }
    None
}

fn cgroup_memory() -> Option<(u64, u64)> {
    for (current_path, limit_path) in [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ] {
        let Some(current) = read_u64(current_path) else {
            continue;
        };
        let Ok(limit_text) = fs::read_to_string(limit_path) else {
            continue;
        };
        let Ok(limit) = limit_text.trim().parse::<u64>() else {
            continue;
        };
        if limit > 0 && limit < (1_u64 << 60) {
            return Some((current, limit));
        }
    }
    None
}

fn file_descriptor_usage() -> Option<(u64, u64)> {
    let open = fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    let limits = fs::read_to_string("/proc/self/limits").ok()?;
    let limit = limits.lines().find_map(|line| {
        let rest = line.strip_prefix("Max open files")?.trim();
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some((open, limit))
}

fn ephemeral_port_usage() -> Option<(u64, u64)> {
    let range = fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range").ok()?;
    let mut values = range
        .split_whitespace()
        .filter_map(|value| value.parse::<u16>().ok());
    let start = values.next()?;
    let end = values.next()?;
    let total = u64::from(end.saturating_sub(start)) + 1;
    let mut ports = std::collections::HashSet::new();
    for path in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(table) = fs::read_to_string(path) else {
            continue;
        };
        for line in table.lines().skip(1) {
            let Some(local) = line.split_whitespace().nth(1) else {
                continue;
            };
            let Some(port_hex) = local.rsplit(':').next() else {
                continue;
            };
            let Ok(port) = u16::from_str_radix(port_hex, 16) else {
                continue;
            };
            if (start..=end).contains(&port) {
                ports.insert(port);
            }
        }
    }
    Some((ports.len() as u64, total))
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}
