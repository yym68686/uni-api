> Historical Python implementation document. This is not the current Rust architecture or a current runtime capability statement. See [architecture](../architecture.md).

# Ember worker and OAIX terminal-hop observability

`uni-api-ember` emits a worker snapshot every five seconds through the existing
bounded Fugue observability queue. Export failures are fail-open and never
block requests or stream delivery.

Each snapshot is exported in two forms: Prometheus-compatible metric events
and a bounded `worker_runtime_snapshot` `app_events` record. The latter keeps
all per-worker facts queryable when a Fugue installation runs without a
Prometheus remote-write backend; it deliberately excludes the latest profile
body and all request data.

## Worker metrics

The worker snapshot and `/v1/observability/runtime` expose:

- `uniapi_ember_worker_cpu_seconds_total`
- `uniapi_ember_worker_cpu_cores`
- `uniapi_ember_worker_sse_events_total` and `_per_second`
- `uniapi_ember_worker_sse_bytes_total` and `_per_second`
- `uniapi_ember_worker_inflight_requests`
- `uniapi_ember_worker_cpu_seconds_per_sse_mebibyte`

One Uvicorn worker is currently enforced per process. The runtime payload also
contains `worker_id`, PID, and the latest bounded CPU profile. Production image
builds inject the triggering Git commit as `SOURCE_COMMIT`, so every profile is
bound to the exact source revision that generated it.

## OAIX terminal-flush hop

OAIX advertises `X-OAIX-Terminal-Flush-Marker: sse-comment-v1`. After a
successful terminal write and flush it emits one SSE comment containing the
exact local flush attempt/completion timestamps, downstream connection ID, and
terminal wire SHA-256. Ember consumes (does not forward) that comment and only
accepts it after the connection ID and terminal hash match.

The raw observation is
`uniapi_ember_oaix_terminal_flush_to_receive_milliseconds`. A cumulative
histogram is also exported with bounded metric names such as
`..._milliseconds_bucket_le_100`, plus `_count` and `_sum`. Invalid timestamps,
hash mismatches, and missing advertised markers are counters and never become
latency samples.

## Triggered on-CPU profile

When sampled process CPU is at least 0.9 cores for two consecutive snapshots,
Ember starts a ten-second, 5 Hz statistical profile. A helper thread reads
per-thread CPU tick deltas from `/proc/self/task/*/stat` and attributes only
those deltas to bounded Python code-location stacks from `sys._current_frames`.
No locals, request bodies, headers, credentials, or response content are read.
Profiles have a 15-minute cooldown and are exported as `worker_on_cpu_profile`
app events.

Safe environment overrides:

- `WORKER_OBSERVABILITY_SAMPLE_INTERVAL_SECONDS` (1–60)
- `WORKER_CPU_PROFILE_ENABLED`
- `WORKER_CPU_PROFILE_TRIGGER_CORES` (0.1–16)
- `WORKER_CPU_PROFILE_TRIGGER_SAMPLES` (1–12)
- `WORKER_CPU_PROFILE_DURATION_SECONDS` (2–30)
- `WORKER_CPU_PROFILE_SAMPLE_HZ` (1–20)
- `WORKER_CPU_PROFILE_COOLDOWN_SECONDS` (60–86400)
