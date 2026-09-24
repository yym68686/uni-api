//! Reuses upstream HTTP clients by the existing transport settings.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Clone, Default)]
pub(crate) struct ClientPool {
    clients: Arc<Mutex<HashMap<ClientKey, reqwest::Client>>>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ClientKey {
    proxy: Option<String>,
    http1_only: bool,
    connect_timeout_ms: Option<u64>,
}

impl ClientPool {
    pub(crate) async fn get(
        &self,
        proxy: Option<&str>,
        http1_only: bool,
        connect_timeout: Option<Duration>,
    ) -> Result<reqwest::Client, String> {
        let key = ClientKey {
            proxy: proxy.map(str::to_owned),
            http1_only,
            connect_timeout_ms: connect_timeout
                .map(|value| value.as_millis().min(u128::from(u64::MAX)) as u64),
        };
        if let Some(client) = self.clients.lock().await.get(&key).cloned() {
            return Ok(client);
        }

        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(256)
            .tcp_keepalive(std::time::Duration::from_secs(30));
        if http1_only {
            builder = builder.http1_only();
        }
        if let Some(connect_timeout) = connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }
        if let Some(proxy_url) = proxy.filter(|value| !value.trim().is_empty()) {
            let configured = reqwest::Proxy::all(proxy_url)
                .map_err(|error| format!("invalid upstream proxy: {error}"))?;
            builder = builder.proxy(configured);
        }
        let client = builder
            .build()
            .map_err(|error| format!("upstream client build failed: {error}"))?;
        self.clients.lock().await.insert(key, client.clone());
        Ok(client)
    }

    pub(crate) async fn len(&self) -> usize {
        self.clients.lock().await.len()
    }
}
