use serde_json::{Map, Value};
use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HedgingConfig {
    pub(crate) enabled: bool,
    pub(crate) max_inflight_attempts: usize,
    pub(crate) winner_policy: WinnerPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WinnerPolicy {
    FirstValidSuccess,
}

impl HedgingConfig {
    pub(crate) fn active(self) -> bool {
        self.enabled
            && self.max_inflight_attempts > 1
            && self.winner_policy == WinnerPolicy::FirstValidSuccess
    }
}

impl Default for HedgingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_inflight_attempts: 1,
            winner_policy: WinnerPolicy::FirstValidSuccess,
        }
    }
}

pub(crate) fn parse_hedging(preferences: &Map<String, Value>) -> HedgingConfig {
    let Some(value) = preferences.get("hedging").and_then(Value::as_object) else {
        return HedgingConfig::default();
    };
    let max_inflight_attempts = value
        .get("max_inflight_attempts")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1)
        .clamp(1, 4);
    let winner_policy = match value
        .get("winner_policy")
        .and_then(Value::as_str)
        .unwrap_or("first_valid_success")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "first_valid_success" => WinnerPolicy::FirstValidSuccess,
        _ => return HedgingConfig::default(),
    };
    HedgingConfig {
        enabled: value
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        max_inflight_attempts,
        winner_policy,
    }
}

#[derive(Clone)]
pub(crate) struct HedgeTrigger<K> {
    key: K,
    sender: mpsc::UnboundedSender<K>,
    fired: Arc<AtomicBool>,
}

impl<K: Clone> HedgeTrigger<K> {
    pub(crate) fn fire(&self) -> bool {
        if self.fired.swap(true, Ordering::AcqRel) {
            return false;
        }
        self.sender.send(self.key.clone()).is_ok()
    }
}

pub(crate) enum HedgeEvent<K, S, F> {
    Triggered { key: K },
    Succeeded { key: K, output: S },
    Failed { key: K, failure: F },
}

pub(crate) struct HedgeScheduler<K, S, F> {
    max_inflight: usize,
    trigger_sender: mpsc::UnboundedSender<K>,
    trigger_receiver: mpsc::UnboundedReceiver<K>,
    result_sender: mpsc::UnboundedSender<(K, Result<S, F>)>,
    result_receiver: mpsc::UnboundedReceiver<(K, Result<S, F>)>,
    running: HashMap<K, tokio::task::JoinHandle<()>>,
}

impl<K, S, F> HedgeScheduler<K, S, F>
where
    K: Clone + Eq + Hash + Send + 'static,
    S: Send + 'static,
    F: Send + 'static,
{
    pub(crate) fn new(max_inflight: usize) -> Self {
        let (trigger_sender, trigger_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        Self {
            max_inflight: max_inflight.max(1),
            trigger_sender,
            trigger_receiver,
            result_sender,
            result_receiver,
            running: HashMap::new(),
        }
    }

    pub(crate) fn has_capacity(&self) -> bool {
        self.running.len() < self.max_inflight
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.running.is_empty()
    }

    pub(crate) fn spawn<Operation, Fut>(&mut self, key: K, operation: Operation)
    where
        Operation: FnOnce(HedgeTrigger<K>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<S, F>> + Send + 'static,
    {
        debug_assert!(self.has_capacity());
        let trigger = HedgeTrigger {
            key: key.clone(),
            sender: self.trigger_sender.clone(),
            fired: Arc::new(AtomicBool::new(false)),
        };
        let result_sender = self.result_sender.clone();
        let result_key = key.clone();
        let handle = tokio::spawn(async move {
            let result = operation(trigger).await;
            let _ = result_sender.send((result_key, result));
        });
        if let Some(previous) = self.running.insert(key, handle) {
            previous.abort();
        }
    }

    pub(crate) async fn next_event(&mut self) -> HedgeEvent<K, S, F> {
        loop {
            tokio::select! {
                biased;
                Some((key, result)) = self.result_receiver.recv() => {
                    self.running.remove(&key);
                    return match result {
                        Ok(output) => HedgeEvent::Succeeded { key, output },
                        Err(failure) => HedgeEvent::Failed { key, failure },
                    };
                }
                Some(key) = self.trigger_receiver.recv() => {
                    if self.running.contains_key(&key) {
                        return HedgeEvent::Triggered { key };
                    }
                }
            }
        }
    }

    pub(crate) fn cancel_remaining(&mut self) -> Vec<K> {
        self.running
            .drain()
            .map(|(key, handle)| {
                handle.abort();
                key
            })
            .collect()
    }
}

impl<K, S, F> Drop for HedgeScheduler<K, S, F> {
    fn drop(&mut self) {
        for (_, handle) in self.running.drain() {
            handle.abort();
        }
    }
}

pub(crate) fn deadline(
    started: tokio::time::Instant,
    seconds: Option<f64>,
) -> Option<tokio::time::Instant> {
    crate::routing::timeouts::positive_duration(seconds)
        .and_then(|duration| started.checked_add(duration))
}

pub(crate) fn earlier_deadline(
    first: Option<tokio::time::Instant>,
    second: Option<tokio::time::Instant>,
) -> Option<tokio::time::Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

pub(crate) async fn await_deadline<F, T>(
    future: F,
    deadline: Option<tokio::time::Instant>,
) -> Result<T, &'static str>
where
    F: Future<Output = T>,
{
    if let Some(deadline) = deadline {
        tokio::time::timeout_at(deadline, future)
            .await
            .map_err(|_| "upstream deadline exceeded")
    } else {
        Ok(future.await)
    }
}

pub(crate) struct HedgeWait<T> {
    pub(crate) output: T,
}

pub(crate) async fn await_with_hedge<F, T, K>(
    future: F,
    soft_deadline: Option<tokio::time::Instant>,
    hard_deadline: Option<tokio::time::Instant>,
    trigger: Option<&HedgeTrigger<K>>,
) -> Result<HedgeWait<T>, &'static str>
where
    F: Future<Output = T>,
    K: Clone,
{
    tokio::pin!(future);
    let can_trigger = trigger.is_some()
        && soft_deadline.is_some_and(|soft| hard_deadline.is_none_or(|hard| soft < hard));
    if can_trigger {
        tokio::select! {
            output = &mut future => return Ok(HedgeWait { output }),
            _ = tokio::time::sleep_until(soft_deadline.expect("checked soft deadline")) => {}
        }
        let _ = trigger.is_some_and(HedgeTrigger::fire);
        let output = await_deadline(&mut future, hard_deadline).await?;
        return Ok(HedgeWait { output });
    }
    let output =
        await_deadline(&mut future, earlier_deadline(soft_deadline, hard_deadline)).await?;
    Ok(HedgeWait { output })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use serde_json::json;

    use super::*;

    #[test]
    fn preferences_parse_with_safe_defaults() {
        let config = parse_hedging(&Map::from_iter([(
            "hedging".into(),
            json!({
                "enabled": true,
                "max_inflight_attempts": 2,
                "winner_policy": "first_valid_success"
            }),
        )]));
        assert!(config.active());
        assert_eq!(config.max_inflight_attempts, 2);

        let invalid = parse_hedging(&Map::from_iter([(
            "hedging".into(),
            json!({"enabled": true, "winner_policy": "unknown"}),
        )]));
        assert_eq!(invalid, HedgingConfig::default());
    }

    #[tokio::test]
    async fn soft_deadline_keeps_original_attempt_alive_until_a_winner_exists() {
        let mut scheduler = HedgeScheduler::<String, &'static str, ()>::new(2);
        scheduler.spawn("slow".into(), |trigger| async move {
            let started = tokio::time::Instant::now();
            await_with_hedge(
                tokio::time::sleep(Duration::from_millis(40)),
                Some(started + Duration::from_millis(5)),
                Some(started + Duration::from_millis(100)),
                Some(&trigger),
            )
            .await
            .unwrap();
            Ok("slow")
        });

        match scheduler.next_event().await {
            HedgeEvent::Triggered { key } => assert_eq!(key, "slow"),
            _ => panic!("soft deadline did not trigger hedging"),
        }
        assert!(!scheduler.is_empty());

        scheduler.spawn("fast".into(), |_| async move { Ok("fast") });
        match scheduler.next_event().await {
            HedgeEvent::Succeeded { key, output } => {
                assert_eq!(key, "fast");
                assert_eq!(output, "fast");
            }
            _ => panic!("fast attempt did not win"),
        }
        assert_eq!(scheduler.cancel_remaining(), vec!["slow".to_owned()]);
    }

    #[tokio::test]
    async fn original_attempt_can_win_after_triggering_a_hedge() {
        let mut scheduler = HedgeScheduler::<String, &'static str, ()>::new(2);
        scheduler.spawn("late".into(), |trigger| async move {
            let started = tokio::time::Instant::now();
            await_with_hedge(
                tokio::time::sleep(Duration::from_millis(20)),
                Some(started + Duration::from_millis(5)),
                Some(started + Duration::from_millis(100)),
                Some(&trigger),
            )
            .await
            .unwrap();
            Ok("late-success")
        });

        match scheduler.next_event().await {
            HedgeEvent::Triggered { key } => assert_eq!(key, "late"),
            _ => panic!("soft deadline did not trigger hedging"),
        }
        match scheduler.next_event().await {
            HedgeEvent::Succeeded { key, output } => {
                assert_eq!(key, "late");
                assert_eq!(output, "late-success");
            }
            _ => panic!("original attempt did not remain eligible to win"),
        }
    }

    #[tokio::test]
    async fn hard_deadline_before_soft_deadline_does_not_trigger_hedging() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let trigger = HedgeTrigger {
            key: "attempt",
            sender,
            fired: Arc::new(AtomicBool::new(false)),
        };
        let started = tokio::time::Instant::now();
        let result = await_with_hedge(
            tokio::time::sleep(Duration::from_millis(50)),
            Some(started + Duration::from_millis(30)),
            Some(started + Duration::from_millis(5)),
            Some(&trigger),
        )
        .await;

        assert_eq!(result.err(), Some("upstream deadline exceeded"));
        assert_eq!(receiver.try_recv(), Err(mpsc::error::TryRecvError::Empty));
    }

    #[tokio::test]
    async fn dropping_scheduler_aborts_detached_attempts() {
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let mut scheduler = HedgeScheduler::<String, (), ()>::new(1);
        let task_flag = dropped.clone();
        scheduler.spawn("pending".into(), move |_| async move {
            let _guard = DropFlag(task_flag);
            std::future::pending::<Result<(), ()>>().await
        });
        tokio::task::yield_now().await;
        drop(scheduler);
        tokio::task::yield_now().await;
        assert!(dropped.load(Ordering::Acquire));
    }
}
