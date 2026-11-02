# Bounded unknown upstream 400 rescue

`preferences.unknown_upstream_400_retries` is a **source-global**, opt-in integer
from `0` to `3` (default `0`, disabled). It is hot-updatable through the reserved
`__uni_console_global_preferences__` settings scope and participates in atomic
validation, encrypted console retention and bootstrap restore. It is not a
provider or caller-key setting. New requests capture the policy; in-flight
requests keep their original budget. No production setting is enabled by a code
release.

The allowance covers additional dispatches authorized by an otherwise terminal,
unknown upstream HTTP400 or uncommitted SSE semantic400. It does **not** increase
the caller key's `AUTO_RETRY` total attempt budget, targeted request limit or
existing hard cap. Existing exact classifiers and proven history repairs run
first and retain their existing budgets and status mappings. An ordinary 500
between unknown400s does not refill this request's rescue allowance.

This first version is deliberately conservative: an actual structured error
envelope with a nonempty message, absent/null/generic code, and no field `param`
is required. Known validation codes/messages, safety/refusal errors and structured
validation details are excluded. Truncated JSON, flattened diagnostic messages
and unstructured text are not enough evidence. SSE must fail before business
output and downstream commitment; opaque streams' existing immediate-commit
contract is unchanged. Successful HTTP200 bodies are not reinterpreted by this
policy.

The exact message `Invalid request parameter` alone is generic, despite containing
the word `invalid`: it identifies no field or constraint. It follows this
configured 0–3 allowance when the same envelope guards pass. A nonempty `param`,
specific code, validation details, or added field/error text still prevents rescue.
This does not introduce an always-on exact rule or change exhaustion from 400.

Only text inference endpoints are eligible. Known hosted tools, explicit
storage/background jobs, server-side conversation references, image/video and
other side-effect APIs are excluded. Both the incoming payload and final compiled
wire request are checked, including provider overrides. This is not an
idempotency guarantee: an upstream may execute work before reporting failure.
Operators accepting this risk can choose a small allowance at the cost of extra
latency, load and possible billing.

The original logical request is retained; this policy adds no payload repair,
truncation, field removal or encrypted-history transformation. Each fallback
continues through the existing per-channel compiler. Failed attempts retain raw
HTTP/billing status (400 for HTTP, 200 for SSE/semantic400); unknown errors remain
400 for routing/final exhaustion, rather than being falsely reclassified as502.
No global provider/key cooldown is introduced by an unknown400.

Only an actual scheduled outbound attempt consumes rescue credit. Cooling,
missing-key, invalid-build and duplicate-backend skips do not. Once rescuing, the
same host+effective port is not revisited under a different provider name/key.
Distinct DNS aliases cannot in general be proven to share a backend; this is a
conservative best-effort identity, not network topology discovery. Speculative
hedges cannot launch additional rescue requests without another failure
authorization, and cannot consume the remaining route cursor while waiting.

The `unknown_upstream_400_retry` event records request/attempt IDs and limit/used
counts only. Final timing summaries include `unknown_400_retry_used`. Never put
request bodies, credentials, encrypted history or full errors into this event.

Deploy a supporting gateway first, then the console editor. A gateway advertises
`unknown_upstream_400_retries` in its settings schema and sends
`X-Uni-Unknown-Upstream-400-Retries: 1` during retained-control restore. The console
must reject unsupported gateways restoring nonzero intent, instead of silently
dropping the policy; it must preserve the previous saved configuration/LKG.

Regression: `tests/http/verify_unknown_400_retry.py` uses loopback servers on
distinct ports, never model providers. It verifies 0–3 credits, total retry cap,
same-backend aliases, hot updates, known-rule compatibility, validation/refusal
exclusions, SSE and collectors, post-output guards, hedging, unchanged wire and
raw billing/final facts. No customer request replay is needed for rollout.
