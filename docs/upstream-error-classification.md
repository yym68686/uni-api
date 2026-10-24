# Generic upstream errors reported as HTTP 400

Some upstream gateways return HTTP 400 with this error envelope:

```json
{"error":{"code":"upstream_error","message":"upstream_error","type":"upstream_error"}}
```

uni-api treats this as an upstream failure, maps it to **502 Bad Gateway**, and
tries the next channel when retries are enabled and budget remains. The existing
`Upstream request failed` message has the same policy. Matching ignores case and
surrounding whitespace; code may be absent, null, or `upstream_error`. A specific
validation code, a non-null parameter, or a message merely quoting the phrase
does not qualify. Ordinary client validation errors retain HTTP 400.

HTTP error envelopes may be wrapped in the existing bounded `error.message` or
`detail.message` JSON format. Responses `error` and `response.failed` events use
the same classification before their structured error is reduced to a message.
An explicit stream error status of 400 becomes 502; other explicit error statuses
retain their meaning. A qualifying generic event without a status also uses 502.

Disabling retries prevents another upstream request but does not turn an upstream
failure back into 400. Exhausted retry budgets return 502. Existing retry limits,
targeted-channel rules and failed-channel cooldown still apply. The upstream HTTP
attempt ledger keeps the actual wire status (400 for HTTP errors, 200 for an SSE
response); routing and final failure facts use the classified status.

Once a stream has sent HTTP headers, its HTTP status cannot change. An error
before business output may still fail over within the existing streaming rules;
an exhausted stream emits the existing in-band error with status 502. After
business output, the gateway preserves the upstream failure event, records the
classified failure, and never replays the request on another channel.

Regression coverage is in `tests/http/verify_minimum_input_retry.py` and the Rust
Responses tests, including the incident sequence 502 → generic 400 → healthy
fallback, disabled retries, bounded retries, exhaustion, cooldown, validation
errors, stream errors, and output already committed.

## Signed thinking history rejected by a Messages provider

An upstream HTTP 400 with `error.type=invalid_request_error` and exactly
`messages.<index>.content.<index>: Invalid \`signature\` in \`thinking\` block`
can be specific to a provider's handling of signed history. For an actual
`/v1/messages` wire request, uni-api verifies that the named assistant content
block is `thinking`, has string thinking content and a nonblank string signature.
Only then is the routing status **502**, allowing the existing next-channel
policy to run. The original request and all signatures/tool records stay intact.

This rule accepts absent/null `code` (or `invalid_request_error`) and absent/null
`param`, including bounded JSON error-message wrappers. It does not cover missing
or malformed signatures, unrelated fields, explicit validation codes, modified
thinking text, prefix-binding explanations, other wire endpoints, or errors
inside an already accepted stream. There is no new post-output retry.

The upstream attempt and billing facts retain **400**. A
`thinking_signature_fallback` event records `attempt_status_code=400` and
`routing_status_code=502` without copying signed content. Disabled retries and
exhausted budgets return 502 with the original error body. Because the failure is
specific to this request's history, it does not cool down the provider or its key;
retry limits and channel order remain unchanged.

The isolated `verify_thinking_signature_retry.py` regression checks unchanged
wire bodies, streaming/nonstreaming fallback, disabled/exhausted retries, strict
validation boundaries, raw/final facts, no channel cooldown and no post-output
retry. It never calls a real model provider.

### Reindexed relay signature errors

A separate narrow branch recognizes the observed relay envelope with
`error.type="<nil>"` and the exact signature message followed by
` (luban request id: req_<id>) (request id: <id>)`. Both IDs must be nonempty,
at most 128 ASCII letters/digits/underscores/hyphens (including `req_` for the
first), with no extra prose. Code and parameter must be absent or null.

This relay reports positions after an internal transformation: the reported
`messages.3.content.79` did not exist in the original Messages body, which another
provider accepted unchanged. For this envelope only, the gateway therefore
checks for actual assistant thinking history with string thinking and nonblank
string signatures instead of interpreting the reported indices. At least one
such block is required; any thinking block with a malformed/missing signature,
non-string thinking, or non-assistant role keeps the client-error policy.

The same 502 routing, raw 400 evidence, retry budget and no-global-cooldown policy
applies. The request is never modified. Standard `invalid_request_error` messages
retain their exact location check. Other `<nil>` errors, missing attribution
suffixes, quoted text, prefix-binding guidance and non-Messages wire errors do
not match. Existing nested error wrappers cannot override a specific validation
code or parameter. This is not a blanket retry for arbitrary HTTP 400 errors.

### Masked relay signature errors

A second observed HTTP 400 envelope uses `error.type="<nil>"` and the exact
message `***.***.content.0: Invalid \`signature\` in \`thinking\` block`
followed by one ` (request id: <opaque-id>)` suffix. Only this literal mask and
block index are accepted; the receipt uses the same bounded opaque-ID grammar.
`code` and `param` must be absent/null. Numeric positions with only one receipt,
other masks, extra prose, and specific validation codes retain their behavior.

Because the masked position cannot locate an original block, this envelope uses
the same complete signed-history validation as the reindexed relay. Requests
and signatures remain unchanged. Raw attempt/billing status stays 400; routing
uses 502 with the existing retry budget, no global cooldown, and no retry after
output has been committed. This does not change ordinary parameter validation.
