use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::sync::Mutex;

const DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

#[derive(Clone)]
pub struct CodexOAuthManager {
    cache: Arc<Mutex<HashMap<String, CachedToken>>>,
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    persisted: Arc<Mutex<Option<HashMap<String, String>>>>,
    store_path: Arc<PathBuf>,
    token_url: Arc<str>,
    client_id: Arc<str>,
    refresh_skew: Duration,
}

#[derive(Clone)]
struct CachedToken {
    access_token: String,
    refresh_token: String,
    expires_at: Option<u64>,
}

pub struct CodexAuth {
    pub bearer: String,
    pub account_id: Option<String>,
}

impl CodexOAuthManager {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            locks: Arc::new(Mutex::new(HashMap::new())),
            persisted: Arc::new(Mutex::new(None)),
            store_path: Arc::new(PathBuf::from(
                std::env::var("CODEX_REFRESH_TOKEN_STORE_PATH")
                    .unwrap_or_else(|_| "./data/codex_refresh_tokens.json".into()),
            )),
            token_url: std::env::var("CODEX_OAUTH_TOKEN_URL")
                .unwrap_or_else(|_| DEFAULT_TOKEN_URL.into())
                .into(),
            client_id: std::env::var("CODEX_OAUTH_CLIENT_ID")
                .unwrap_or_else(|_| DEFAULT_CLIENT_ID.into())
                .into(),
            refresh_skew: Duration::from_secs(30),
        }
    }

    pub async fn resolve(&self, raw_key: &str, proxy: Option<&str>) -> Result<CodexAuth, String> {
        let raw = raw_key.trim();
        let Some((account_id, configured_refresh)) = split_refresh_key(raw)? else {
            return Ok(CodexAuth {
                bearer: raw.to_owned(),
                account_id: None,
            });
        };
        let lock = {
            let mut locks = self.locks.lock().await;
            locks
                .entry(raw.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = lock.lock().await;
        if let Some(token) = self.cache.lock().await.get(raw).cloned() {
            if token_is_valid(&token, self.refresh_skew) {
                return Ok(CodexAuth {
                    bearer: token.access_token,
                    account_id,
                });
            }
        }
        let persisted = self.persisted_refresh_token(raw).await;
        let refresh_token = self
            .cache
            .lock()
            .await
            .get(raw)
            .map(|token| token.refresh_token.clone())
            .or(persisted)
            .unwrap_or(configured_refresh);
        let mut builder = reqwest::Client::builder().http1_only();
        if let Some(proxy) = proxy.filter(|value| !value.trim().is_empty()) {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy)
                    .map_err(|error| format!("invalid Codex OAuth proxy: {error}"))?,
            );
        }
        let client = builder
            .build()
            .map_err(|error| format!("build Codex OAuth client: {error}"))?;
        let refreshed = self.refresh(&client, &refresh_token).await?;
        self.persist_refresh_token(raw, &refreshed.refresh_token)
            .await;
        self.cache
            .lock()
            .await
            .insert(raw.to_owned(), refreshed.clone());
        Ok(CodexAuth {
            bearer: refreshed.access_token,
            account_id,
        })
    }

    pub async fn clear(&self, raw_key: &str) {
        self.cache.lock().await.remove(raw_key);
    }

    async fn refresh(
        &self,
        client: &reqwest::Client,
        refresh_token: &str,
    ) -> Result<CachedToken, String> {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", self.client_id.as_ref())
            .append_pair("grant_type", "refresh_token")
            .append_pair("refresh_token", refresh_token)
            .append_pair("scope", "openid profile email")
            .finish();
        let response = client
            .post(self.token_url.as_ref())
            .header("content-type", "application/x-www-form-urlencoded")
            .header("accept", "application/json")
            .timeout(Duration::from_secs(30))
            .body(body)
            .send()
            .await
            .map_err(|error| format!("Codex token refresh request failed: {error}"))?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|error| format!("Read Codex token refresh response failed: {error}"))?;
        if !status.is_success() {
            return Err(format!(
                "Codex token refresh failed: status {}: {}",
                status.as_u16(),
                String::from_utf8_lossy(&body)
            ));
        }
        let payload: Value = serde_json::from_slice(&body)
            .map_err(|error| format!("Decode Codex token refresh response failed: {error}"))?;
        let access_token = payload
            .get("access_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .ok_or_else(|| "Codex token refresh returned empty access_token".to_owned())?
            .to_owned();
        let refreshed = payload
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .unwrap_or(refresh_token)
            .to_owned();
        let expires_at = payload
            .get("expires_in")
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
            })
            .filter(|seconds| *seconds > 0)
            .map(|seconds| unix_seconds().saturating_add(seconds));
        Ok(CachedToken {
            access_token,
            refresh_token: refreshed,
            expires_at,
        })
    }

    async fn persisted_refresh_token(&self, raw_key: &str) -> Option<String> {
        self.ensure_store_loaded().await;
        self.persisted
            .lock()
            .await
            .as_ref()
            .and_then(|tokens| tokens.get(raw_key).cloned())
    }

    async fn persist_refresh_token(&self, raw_key: &str, refresh_token: &str) {
        self.ensure_store_loaded().await;
        let encoded = {
            let mut persisted = self.persisted.lock().await;
            let values = persisted.get_or_insert_with(HashMap::new);
            if values
                .get(raw_key)
                .is_some_and(|value| value == refresh_token)
            {
                return;
            }
            values.insert(raw_key.to_owned(), refresh_token.to_owned());
            match serde_json::to_vec_pretty(values) {
                Ok(encoded) => encoded,
                Err(_) => return,
            }
        };
        if let Some(parent) = self.store_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let temporary = self
            .store_path
            .with_extension(format!("tmp.{}", std::process::id()));
        if tokio::fs::write(&temporary, encoded).await.is_err() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600))
                .await;
        }
        let _ = tokio::fs::rename(&temporary, self.store_path.as_ref()).await;
    }

    async fn ensure_store_loaded(&self) {
        let mut persisted = self.persisted.lock().await;
        if persisted.is_some() {
            return;
        }
        let values = tokio::fs::read(self.store_path.as_ref())
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice::<HashMap<String, String>>(&bytes).ok())
            .unwrap_or_default();
        *persisted = Some(values);
    }
}

fn split_refresh_key(raw: &str) -> Result<Option<(Option<String>, String)>, String> {
    let Some((account_id, refresh_token)) = raw.split_once(',') else {
        return Ok(None);
    };
    let refresh_token = refresh_token.trim();
    if refresh_token.is_empty() {
        return Err("Invalid Codex API key format: expected 'account_id,refresh_token'".into());
    }
    Ok(Some((
        (!account_id.trim().is_empty()).then(|| account_id.trim().to_owned()),
        refresh_token.to_owned(),
    )))
}

fn token_is_valid(token: &CachedToken, skew: Duration) -> bool {
    token
        .expires_at
        .is_none_or(|expires_at| unix_seconds() < expires_at.saturating_sub(skew.as_secs()))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_static_and_refresh_credentials() {
        assert!(split_refresh_key("token").unwrap().is_none());
        let (account, token) = split_refresh_key("acct, refresh").unwrap().unwrap();
        assert_eq!(account.as_deref(), Some("acct"));
        assert_eq!(token, "refresh");
        assert!(split_refresh_key("acct,").is_err());
    }
}
