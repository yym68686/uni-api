# Responses empty tool-name repair

Native `/v1/responses` can recover a rejected history containing an empty
function name when its paired output explicitly says the tool was not executed.
The original request is sent unchanged. A qualifying HTTP 400 triggers one
same-channel resend with only the confirmed historical pairs converted to text.

## Exact trigger

Repair requires `kind=http_error`, both HTTP status fields equal to 400, and
`committed=false`. The official error must have all of:

```json
{"error":{"code":"empty_string","type":"invalid_request_error","param":"input[82].name","message":"Invalid 'input[82].name': empty string. Expected a string with minimum length 1, but got an empty string instead."}}
```

Any nonnegative decimal index is accepted; the index in `param` and the exact
message must agree. It is not used to index the original request, since upstream
gateways may prepend or transform input. The implementation also accepts the two
observed rewrites of this exact message:

- `error.type=upstream_error`, with no `code` or `param`.
- `error.type=gateway_error`, `code=oaix_gateway_error`, `status=400`, no `param`.

Up to two untyped `error.message` JSON wrappers are supported. A single SSE
`event: error` frame with one JSON `data:` line is accepted, including inside a
wrapper. Additional SSE events, substring matches, arbitrary text, mismatched
validation fields and request/debug echoes do not activate repair.

## Body transformation

A pair qualifies only if:

- The call is `function_call`, `name` is exactly `""`, and `arguments` is a string.
- Its `call_id` is a nonblank string used by exactly two input items.
- The later item is `function_call_output` with output exactly
  `"unsupported call: "` (including the trailing space).

All qualifying pairs are converted in place. The call becomes an assistant
message with `output_text`: `Historical malformed tool call (not executed): `
followed by the complete original call JSON. Its result becomes a user message
with `input_text`: `Historical tool execution error: ` followed by the complete
original output JSON. Arguments, call ID and error context are preserved as text;
no function name is guessed. Valid calls, executed outputs, reasoning and other
request fields remain unchanged. Missing, duplicate and reversed pairs are not
repaired.

## Retry behavior

The resend preserves the exact URL, upstream key, model, proxy and request
options. It has its own attempt ID, dispatch measurement and upstream result.
There is at most one historical-context repair resend per incoming request,
shared with heartbeat, [missing reasoning-item repair](responses-missing-item-repair.md)
and [rejected encrypted-envelope repair](responses-encrypted-content-repair.md),
including under hedging.

If the empty-name repair resend still returns HTTP 400 before commitment, the
configured next-channel retry continues even though ordinary parameter 400s
normally stop retries. Other failures follow normal retry classification.
Subsequent channels receive their own compilation of the **original** input;
the repaired body is local to its attempt. If those channels reject the same
confirmed bad history, they can be skipped within the existing routing budget,
without a second repair resend. Unrelated parameter 400s on a later channel
still stop retries. The original status and error remain in upstream facts;
exhaustion returns the actual last failure, not a fabricated success.

`AUTO_RETRY=false`, administrator-targeted diagnostics,
`/v1/responses/compact`, and failures after committed business output do not
activate repair. Configuration, persisted conversations, credentials and other
channels' payloads are not modified.

The `responses_empty_name_repair` log event contains the request ID, original and
repair attempt IDs, provider/model and `tool_pairs_converted`, with no request
text or credentials. Existing heartbeat log fields are unchanged.

## Verification

Unit tests cover strict recognition, known wrappers, negative matches, pair
identity, original-item preservation and idempotence. The isolated HTTP suite
`tests/http/verify_empty_name_retry.py` covers successful repair, same-key retention
with rotating keys, retry after repeated/other 400 and 503, multi-channel
exhaustion, normal-request preservation, opt-outs, postcommit errors and a hedge
race. `tests/http/verify_heartbeat_retry.py` checks the existing repair behavior.
