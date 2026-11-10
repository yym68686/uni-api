# Shared request execution contract

Native Responses and generic execution consume the same request-scoped inputs
and decisions. The native path retains its Responses stream transport; generic
execution retains protocol conversion and other endpoint transports.

* `GatewayRuntime::for_request` fixes the effective configuration snapshot and
  routing rules before admission. Its clone shares scheduling and live controls
  with the process. A configuration reload affects the next request, including
  when an existing request is waiting for a retry.
* `RequestFacts` contains the normalized model, endpoint, body size, request type,
  reasoning effort and downstream stream intent. `resolve_route` authenticates,
  checks balance, atomically admits global/client/nested rate buckets, filters
  candidates and schedules one `ResolvedRoute` inside `RequestContext`.
* If native preparation selects generic execution, it hands over that context
  and the normalized payload. It does not repeat admission or advance scheduling
  again. An internal moderation receipt prevents repeating moderation.
* Every attempt checks current channel disable rules, channel/key cooldown and
  provider-key rate limits. These facts remain dynamic. An emergency disable
  prevents a later dispatch; it does not cancel an already dispatched request.
* `ResolvedAttemptPolicy` resolves proxy, HTTP version, timeouts, keepalive and
  Responses commit/normalization limits from the fixed snapshot. Codex OAuth
  refresh uses the same proxy selection. `providers::responses` constructs the
  shared Responses body and headers; Chat conversion keeps its own defaults.
* `FinalFacts` distinguishes protocol terminal, semantic status, upstream HTTP
  status, downstream HTTP status and business commit. Missing evidence remains
  unknown. Responses completion requires a validated successful terminal event.
  HTTP 200, EOF, usage and first output do not by themselves prove completion.
* `ReplayCapture` bounds capture memory for both paths. A validated complete
  response may be replayed; an incomplete, interrupted or oversized response
  keeps an executed/nonreplayable claim. Body EOF cannot authorize reexecution.
  Generic live streams supply completion through the protocol observer;
  synthetic streams supply it after their terminal payload has been built.

`tests/http/verify_request_contract.py` exercises native and generic routing,
admission, moderation, Responses wire parity, commit policy, completion facts,
idempotency, reload during fallback and live emergency disable using loopback
fixtures. `scripts/verify-http.py` also runs the existing endpoint regressions.
