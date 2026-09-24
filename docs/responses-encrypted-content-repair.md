# Responses rejected encrypted-envelope repair

Native `/v1/responses` can recover visible summaries from a specific non-native
`cursor-sand-v1:` envelope rejected by an upstream. The initial request is
unchanged. This is a reactive compatibility fallback, not decryption or recovery
of opaque native reasoning.

## Exact trigger

The outcome must be `http_error`, both status fields must be 400, and
`committed` must explicitly be false. The structured JSON error must match:

```json
{"error":{"code":"invalid_encrypted_content","type":"invalid_request_error","param":null,"message":"The encrypted content for item rs_example could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}
```

Only the ID varies: `rs_` followed by ASCII alphanumeric characters. Up to two
untyped `error.message` JSON wrappers are supported. Missing or contradictory
fields, quoted messages, request echoes, arbitrary text and other errors are
not triggers.

The actual upstream body must have `store: false`, an input array and exactly
one item matching the rejected ID. That target must satisfy every envelope
condition below before any item is converted.

## Eligible envelope and conversion

An eligible item has:

- `type: reasoning` and a unique valid `rs_` ID.
- Absent, null, or empty-array `content`.
- A nonempty list of known `summary_text` entries with string text; at least
  one string must contain non-whitespace text.
- String `encrypted_content` starting with `cursor-sand-v1:`, at most 2 MiB.
- Strict standard Base64 after the prefix, decoding to JSON with exactly
  `signature` and `text` string fields. Duplicate or unknown envelope fields
  are rejected. The signature must be nonempty.
- Decoded `text` exactly equal to all visible summary strings, in order,
  joined by two newline characters.

After confirming the target, all items in this request that independently meet
these same conditions become ordinary assistant messages:

```json
{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Historical reasoning summary:\nFirst summary.\n\nSecond summary."}]}
```

This bounded batch matters: an observed history contained three adjacent
envelopes, and fixing only the first made the upstream reject the second. All
other input items and top-level request fields stay unchanged. Native encrypted
content, unknown envelopes, mismatched text, duplicate IDs, bare references and
unrecognized summaries are never silently deleted or converted. No text is
promoted to a system or developer instruction.

Live synthetic comparisons on the affected channel reproduced the exact 400.
Deleting only `encrypted_content` caused a missing-item 404. Deleting both the
blob and ID produced HTTP 200 but failed to recall a marker from the summary.
Assistant-message conversion completed and retained the marker, including
when all three eligible envelopes were converted. These observations do not
establish that all native encrypted-content errors can be repaired this way.

## Retry and observations

The same provider, URL, key, model, proxy and options are reused once with a new
attempt ID. This shares the **one historical repair resend per incoming
request** budget with heartbeat, empty-name and missing-item repairs, including
under hedging. The resend does not consume an ordinary provider-selection slot.

If that resend fails with an uncommitted HTTP 400 or 404, normal channel
selection continues within its existing budget. Other failures follow existing
classification. Each later channel independently compiles the original request;
it does not inherit the first provider's overrides or converted input. A later
rejection of the same eligible encrypted history may advance to another channel
without another repair resend. Unrelated later 400s retain their usual stop
behavior. Original upstream statuses remain in the facts, including exhaustion.

`AUTO_RETRY=false`, targeted diagnostics, compact requests and failures after
business output is committed do not trigger this repair. A different historical
error does not open another repair budget.

The `responses_encrypted_content_repair` event contains request ID, original and
repair attempt IDs, provider/model and `reasoning_items_converted`. It never
includes source summaries, envelope contents, item IDs or keys.

Unit tests live in `src/protocols/responses/encrypted_content.rs`; isolated HTTP
tests in `tests/http/verify_encrypted_content_retry.py` cover streaming and
nonstreaming requests, same-key resend, fallback and exhaustion, provider-local
overrides, strict negative cases, cross-repair budgets, postcommit and hedge races.
