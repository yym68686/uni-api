# Generic support-only upstream HTTP 400

A provider HTTP 400 is treated as a gateway failure (502) when its structured
error has the complete message `Request could not be completed. Contact support
with the request ID.`, type `invalid_request_error`, absent/null or generic
`invalid_request_error` code, and no non-null parameter. The opaque upstream
request ID is not part of the match. Existing JSON error-message wrappers are
recognized; echoed request fields and partial message matches are not.

This classification uses the existing next-channel retry policy. With automatic
retry disabled, or after exhausting the configured retry budget, the public
failure status is 502. Ordinary validation errors remain request-scoped 400s.
The original upstream response and attempt status remain available for diagnosis;
classification does not rewrite the request body or replay committed output.

Offline regressions cover Responses (including Codex), Chat Completions, compact,
streaming/nonstreaming, hedging on/off, retries disabled, exhaustion, and
nonmatching validation/echoed errors. Run `cargo test --locked` and
`python3 scripts/verify-http.py` after `cargo build --locked`.
