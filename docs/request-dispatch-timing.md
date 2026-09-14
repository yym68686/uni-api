# Request arrival to channel dispatch

Channel metrics and timeseries rows include `stats.request_to_dispatch`, with
`sample_count`, `mean_ms`, `p50_ms`, `p95_ms`, `last_ms`, and `last_observed_at`.
The frontend shows the window's p50 in the request-before-wait column; its tooltip
includes p95, the latest exact sample, and the dispatch sample count. Quantiles
use histogram upper bounds, as existing first-output metrics do.

The timer starts at the uni-api Rust HTTP handler entry, before request body
reading/decompression, resource admission, request spooling and routing. The
monotonic timestamp is an internal request extension, never a client header.
The same timestamp is carried through fallback, retries and hedged requests.
It stops immediately before calling the model upstream's HTTP `send()`.

It therefore includes body upload, gateway resource wait, moderation preflight,
key/auth and payload preparation, and time spent on earlier attempts. It excludes
0-0, edge and network time before the uni-api handler, HTTP header parsing before
handler entry, and this attempt's HTTP client connection/pool/DNS/TLS/send/response
time after `send()` starts. It is not just scheduler queue time.

Only dispatched model attempts create samples, once per attempt (including cloned
hedged plans). A plan skipped for cooldown, rejected before sending, or an
idempotency replay creates no dispatch sample. Internal moderation preflight is
included in the parent wait but is not counted as a separately arrived request.
Streaming/nonstreaming Responses and generic model attempts are instrumented.

Samples enter their minute bucket at dispatch time, even while the request is
still running or before it fails. Success-rate counts use terminal time, so their
sample counts are not expected to match this distribution in every time window.
The API declares `request_to_dispatch_sample_basis=dispatch_at` and
`request_to_dispatch_measurement=uni_api_handler_entry_to_upstream_http_send`.
Memory retention and platform authentication are unchanged. New instances start
with no samples; historical values are not inferred or backfilled.

Each `channel_dispatch` structured event records the exact
`request_to_dispatch_ms`, `request_id`, `attempt_id`, provider, requested/upstream
model, endpoint and stream flag. It has no request body, credentials or user data.
Use these events to inspect an individual request; p50 is an aggregate across
attempts, not an individual request timeline. No routing order, admission policy,
timeouts, retries or cooldowns are changed by this instrumentation.

Verification:

```sh
cargo test --manifest-path rust/uni-api-native/Cargo.toml --locked
cargo build --manifest-path rust/uni-api-native/Cargo.toml --locked
python3 scripts/verify_dispatch_timing.py rust/uni-api-native/target/debug/uni-api-front
```

The offline integration test launches only local fake upstreams and isolated
gateway instances. It delays body upload and the first attempt, checks dispatch
samples before the second response finishes, and covers streaming/nonstreaming
Responses, generic Chat Completions, idempotent spooling and hedging on both
paths. Test-only resource ledger/spool paths and disk reserve overrides keep it
independent of other local gateway processes and production configuration.
