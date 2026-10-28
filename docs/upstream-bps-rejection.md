# BPS upstream rejection reported as HTTP 400

Only HTTP 400 with a structured `error` (or existing JSON-message wrapper)
matching all three exact fields is classified as a gateway failure:

- `type`: `invalid_request_error`
- `code`: `basispoints_upstream_error`
- `message`: `Excel BPS rejected this request; account scheduling was not changed`

`param` must be absent or null. A code alone, partial/quoted message, different
type, or explicit parameter keeps the normal client-error policy. Other HTTP
statuses are unchanged.

The routing status is 502 and uses the existing bounded next-channel retry and
failed-route cooldown policy. Disabling automatic retries, or exhausting their
budget, returns 502. Original upstream attempt and billing statuses remain 400.
No request fields, encrypted history, tools, or output limits are changed. Output
that has already been committed is never replayed.

Run `cargo test --locked`, `cargo build --locked`, and
`python3 scripts/verify-http.py`. The provider-failure HTTP suite covers native
Responses, Codex Responses, Chat Completions, compact, streaming/nonstreaming,
hedging, retry-disable/exhaustion/budget cases and nonmatching envelopes.
