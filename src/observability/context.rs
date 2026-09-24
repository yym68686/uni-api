use axum::http::HeaderMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn request_id(headers: &HeaderMap) -> String {
    for name in ["x-request-id", "x-caller-request-id"] {
        if let Some(value) = headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return value.chars().take(128).collect();
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    format!("{now:016x}{sequence:016x}")
}
