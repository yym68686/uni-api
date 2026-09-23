# Responses missing reasoning-item repair

Native `/v1/responses` can recover the visible summary of one historical
reasoning item rejected because its ID cannot be retrieved. The first request
is unchanged. Repair is reactive and local to one upstream attempt; it does not
enable storage, mutate configuration, or edit a persisted conversation.

## Exact trigger and eligible input

Both the HTTP status and upstream status must be 404, the outcome must be
`http_error`, and `committed` must explicitly be false. The JSON error must
contain `code: null`, `type: invalid_request_error`, `param: input`, and this exact
message shape (only the reasoning ID varies):

```json
{"error":{"code":null,"type":"invalid_request_error","param":"input","message":"Item with id 'rs_c65ee145dd863116153ca83b' not found. Items are not persisted when `store` is set to false. Try again with `store` set to true, or remove this item from your input."}}
```

The ID must start with `rs_` followed by ASCII alphanumeric characters. Up to
two untyped `error.message` JSON wrappers are accepted. Plain text, request
echoes, missing or contradictory error fields, other IDs and other 404 messages
do not activate this repair.

The actual upstream request body must have `store: false`. Its `input` must
contain exactly one item with the error's ID, and that item must:

- Have `type: reasoning`.
- Have absent, null, or empty-string `encrypted_content`.
- Have absent, null, or empty-array `content`.
- Have a nonempty list of `summary_text` entries with string text, at least one
  of which is not whitespace.

Encrypted reasoning, raw content, duplicate IDs, bare `item_reference` objects,
and empty or unknown summaries are not silently discarded or reinterpreted.

## Transformation

Only the identified input item becomes:

```json
{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Historical reasoning summary:\nFirst summary.\n\nSecond summary."}]}
```

Every summary string is retained in order, separated by two newlines. Other
reasoning items, messages, tool calls, outputs, and top-level request fields are
unchanged. This preserves the summary actually received by the client; it does
not reconstruct missing encrypted reasoning. In live synthetic comparisons,
merely removing the ID produced HTTP 200 but failed to recall a marker present
only in the summary; conversion to an assistant message preserved that marker.

## Retry and observation

The repair resend retains the same provider, URL, key, model, proxy and options,
with a new attempt ID. It shares the **one historical repair resend per incoming
request** limit with heartbeat and empty-name repair, including under hedging.
It does not consume an ordinary provider-selection slot.

If the repair resend fails with an uncommitted HTTP 404 or 400, ordinary channel
selection continues within its existing budget. Other failures use existing
classification. Each later channel compiles the original request independently;
it does not inherit another provider's request overrides or repaired body. A
later rejection of the same safely repairable history can advance to the next
channel without another repair resend. An unrelated parameter 400 on a later
channel retains its normal stop behavior. Exhaustion reports the actual last
failure; upstream facts keep the original HTTP status.

`AUTO_RETRY=false`, administrator-targeted diagnostics, compact requests and
errors after committed business output do not trigger repair. A different
historical error appearing after a repair does not open an additional repair
budget.

The `responses_missing_item_repair` log event records request ID, original and
repair attempt IDs, provider/model and `reasoning_items_converted: 1`. It does
not log the source summary, full input, missing item ID or credentials.

Tests live in `responses_missing_item.rs` and
`scripts/verify_missing_item_retry.py`. The HTTP suite covers strict matching,
same-key preservation, one-item replacement, negative eligibility cases,
streaming/nonstreaming, retry after repair, exhaustion, no cross-repair loops,
postcommit behavior and a hedge race. Existing heartbeat/empty-name regressions
continue to run in CI.
