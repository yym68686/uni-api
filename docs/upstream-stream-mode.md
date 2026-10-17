# Responses upstream streaming capability

Channel preference `upstream_stream_mode` separates the upstream transport from
what the caller requested. The default `follow_client` preserves existing behavior.
Use `stream_only` for a Responses provider that rejects non-streaming requests.

```yaml
preferences:
  upstream_stream_mode:
    default: follow_client
    models:
      deepseek-v4.1-flash: stream_only
```

A string `stream_only` applies to all models on this channel. Model overrides match
exact public model names first, then exact upstream names. This setting applies only
to Responses generation upstreams, including Chat/Gemini callers using the existing
Responses adapter. It does not force streaming on compact, search, embeddings,
Messages or native Gemini upstream endpoints. There is no automatic inference or
persisted mutation after a 400 error. The capability takes precedence over request
body stream overrides on supported generation endpoints.

The caller's absent/false `stream` still returns `application/json`. Collection
validates SSE, uses the complete `response.completed.response` or
`response.incomplete.response`, preserves native output items and extensions, and
keeps incomplete status distinct from success. Failed events, malformed or truncated
streams, missing terminal objects and oversized responses fail explicitly. It does
not reconstruct a partial terminal from deltas or round-trip native Responses through
Chat. Existing cross-protocol adapters retain their existing conversion limits.

Collection buffers at most `RUST_GENERIC_UPSTREAM_RESPONSE_MAX_BYTES` (64 MiB by
default) of received SSE per attempt, honors idle and total deadlines, and returns
when the terminal is read. The collection future owns its upstream connection.
Timeout rules match the caller's stream mode. Failures before semantic output may
use the existing retry budget; failures after semantic output do not initiate a
replacement attempt. Already configured concurrent hedge attempts retain their
normal scheduling. Native observations and generic collection metadata distinguish
upstream and downstream stream modes and retain terminal usage/timing.

The settings schema advertises `upstream_stream_mode: 1`; preview shows both stream
modes. Use the console's revision-checked settings journal, encrypted retention,
and atomic restore. Boot requests advertise `X-Uni-Upstream-Stream-Mode: 1`.
The console rejects restoration of new transport settings into older gateways rather
than silently discarding them; the saved intent and serving instance are preserved.
Deploy gateway support before enabling this configuration. Configuration rollback
uses the existing revision-checked setting removal/reset, independently of releases.

Validation: `tests/http/verify_stream_only.py` exercises native and mixed routes,
Chat conversion, fragmented UTF-8/SSE, tools/reasoning/cache usage, explicit false
and absent stream, streaming calls, incomplete/failed/truncated events, byte and
time limits, retry boundaries, preview, hot changes and startup restoration.
