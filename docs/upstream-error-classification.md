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
