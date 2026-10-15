# Upstream Gemini key platform mismatch

An upstream HTTP 400 with the full structured error `code: 400`,
`status: INVALID_ARGUMENT`, and message `API key group platform is not gemini`
is a provider configuration failure. It maps to routing HTTP 502 and uses the
existing next-channel retry policy. Root, error/detail envelopes and existing
JSON-string message wrappers are supported. Non-null parameter errors, nonempty
details, ordinary INVALID_ARGUMENT errors and echoed input do not match.

Raw upstream status/body remain recorded as received. With AUTO_RETRY disabled
or the retry budget exhausted, the failure returns 502. Committed output is not
replayed, and request bodies are not changed. Verification uses local mock
upstreams, never customer request replay or paid inference.
