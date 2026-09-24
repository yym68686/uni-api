use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode};
use axum::response::IntoResponse;

pub(crate) const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";

pub(crate) const SPOOL_HEADER_PREFIX: &str = "x-uni-api-rust-request-spool-";

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
            || name.as_str().starts_with(SPOOL_HEADER_PREFIX)
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

pub(crate) fn is_hop_by_hop(name: &str) -> bool {
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
