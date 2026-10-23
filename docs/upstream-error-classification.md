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
