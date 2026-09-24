//! Small bounded-compatible replacement for Fugue's Python exporter.

//! Structured events always remain on stderr; when FUGUE_OBSERVABILITY_ENDPOINT

//! is configured they are also sent as JSON to the collector asynchronously.

use serde_json::Value;
use std::sync::OnceLock;

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

pub(crate) fn emit(event: Value) {
    eprintln!("{}", event);
    let Ok(endpoint) = std::env::var("FUGUE_OBSERVABILITY_ENDPOINT") else {
        return;
    };
    let endpoint = endpoint.trim().to_owned();
    if endpoint.is_empty() {
        return;
    }
    let client = CLIENT.get_or_init(reqwest::Client::new).clone();
    tokio::spawn(async move {
        let _ = client
            .post(endpoint)
            .json(&event)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await;
    });
}
