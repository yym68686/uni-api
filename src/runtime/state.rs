use crate::config::snapshot::runtime_provider;
use crate::config::snapshot::ApiKey;
use crate::config::snapshot::RawSnapshot;
use crate::config::snapshot::Snapshot;
use crate::config::snapshot::SNAPSHOT_SCHEMA_VERSION;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct SnapshotStamp {
    pub(crate) modified: Option<SystemTime>,
    pub(crate) len: u64,
}

#[derive(Clone)]
pub struct GatewayRuntime {
    pub(crate) scheduling: crate::runtime::scheduling::SchedulingState,
    pub(crate) channel_controls: Arc<RwLock<crate::control::channels::Controls>>,
    pub(crate) path: Arc<PathBuf>,
    pub(crate) current: Arc<RwLock<Option<Arc<Snapshot>>>>,
    pub(crate) snapshot_stamp: Arc<Mutex<Option<SnapshotStamp>>>,
}

impl GatewayRuntime {
    pub fn new() -> Self {
        let path = std::env::var("RUST_RESPONSES_CONFIG_SNAPSHOT_PATH")
            .unwrap_or_else(|_| "/tmp/uni-api-rust-responses-config-v1.json".into());
        Self {
            scheduling: crate::runtime::scheduling::SchedulingState::default(),
            channel_controls: Arc::new(RwLock::new(crate::control::channels::Controls::default())),
            path: Arc::new(PathBuf::from(path)),
            current: Arc::new(RwLock::new(None)),
            snapshot_stamp: Arc::new(Mutex::new(None)),
        }
    }

    pub fn start_watcher(&self) {
        let store = self.clone();
        tokio::spawn(async move {
            loop {
                if let Err(error) = store.refresh().await {
                    eprintln!(
                        "{}",
                        json!({
                            "event_type": "rust_responses_config_snapshot_error",
                            "error": error,
                        })
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    pub async fn refresh(&self) -> Result<bool, String> {
        let metadata = match tokio::fs::metadata(self.path.as_ref()).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(format!("stat native Responses snapshot: {error}")),
        };
        let stamp = SnapshotStamp {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        };
        if self
            .snapshot_stamp
            .lock()
            .await
            .as_ref()
            .is_some_and(|current| *current == stamp)
        {
            return Ok(false);
        }
        let bytes = match tokio::fs::read(self.path.as_ref()).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(format!("read native Responses snapshot: {error}")),
        };
        let raw: RawSnapshot = serde_json::from_slice(&bytes)
            .map_err(|error| format!("decode native Responses snapshot: {error}"))?;
        if raw.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported native Responses snapshot schema {}",
                raw.schema_version
            ));
        }
        if raw.revision.len() != 64 {
            return Err("native Responses snapshot revision is invalid".into());
        }
        *self.snapshot_stamp.lock().await = Some(stamp);
        if self
            .current
            .read()
            .await
            .as_ref()
            .is_some_and(|snapshot| snapshot.revision.as_ref() == raw.revision)
        {
            return Ok(false);
        }

        let mut cursors = self.scheduling.provider_cursors.lock().await;
        let mut providers = Vec::with_capacity(raw.providers.len());
        let mut providers_by_name = HashMap::with_capacity(raw.providers.len());
        for item in raw.providers {
            let name = item.name.trim().to_owned();
            if name.is_empty() || item.models.is_empty() {
                continue;
            }
            let cursor = cursors
                .entry(name.clone())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
                .clone();
            let provider = runtime_provider(item, cursor);
            providers_by_name.insert(name, provider.clone());
            providers.push(provider);
        }
        drop(cursors);

        let api_key_order = raw
            .api_keys
            .iter()
            .map(|item| item.token.trim().to_owned())
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let api_keys = raw
            .api_keys
            .into_iter()
            .filter_map(|item| {
                let token = item.token.trim().to_owned();
                if token.is_empty() {
                    return None;
                }
                let native_supported = item.model_rules.iter().all(Value::is_string);
                let rules = item
                    .model_rules
                    .into_iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect::<Vec<_>>();
                Some((
                    token.clone(),
                    Arc::new(ApiKey {
                        token: token.into(),
                        model_rules: Arc::new(rules),
                        role: item.role.into(),
                        preferences: Arc::new(item.preferences),
                        weights: Arc::new(item.weights),
                        native_supported,
                    }),
                ))
            })
            .collect::<HashMap<_, _>>();
        let snapshot = Arc::new(Snapshot {
            revision: raw.revision.into(),
            preferences: Arc::new(raw.preferences),
            api_keys: Arc::new(api_keys),
            api_key_order: Arc::new(api_key_order),
            providers: Arc::new(providers),
            providers_by_name: Arc::new(providers_by_name),
            api_config: Arc::new(raw.api_config),
        });
        *self.current.write().await = Some(snapshot.clone());
        eprintln!(
            "{}",
            json!({
                "event_type": "rust_responses_config_snapshot_loaded",
                "revision": snapshot.revision.as_ref(),
                "api_key_count": snapshot.api_keys.len(),
                "provider_count": snapshot.providers.len(),
            })
        );
        Ok(true)
    }

    pub(crate) async fn base_snapshot(&self) -> Option<Arc<Snapshot>> {
        self.current.read().await.clone()
    }

    pub(crate) async fn snapshot(&self) -> Option<Arc<Snapshot>> {
        let base = self.current.read().await.clone()?;
        Some(self.channel_controls.read().await.overlay(base))
    }

    pub async fn is_ready(&self) -> bool {
        self.current.read().await.is_some()
    }

    pub async fn runtime_counts(&self) -> (usize, usize, usize) {
        let Some(snapshot) = self.snapshot().await else {
            return (0, 0, 0);
        };
        let model_count = snapshot
            .providers
            .iter()
            .flat_map(|provider| provider.models.keys())
            .collect::<BTreeSet<_>>()
            .len();
        (
            snapshot.api_keys.len(),
            snapshot.providers.len(),
            model_count,
        )
    }

    pub async fn api_config(&self) -> Option<Value> {
        self.snapshot()
            .await
            .map(|snapshot| (*snapshot.api_config).clone())
    }
}
