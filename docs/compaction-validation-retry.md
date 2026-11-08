# Compaction response validation and retry

Validation applies only to explicit `/v1/responses` inputs containing
`compaction_trigger`, and `/v1/responses/compact`. Ordinary Responses calls keep
their existing commitment and retry behavior.

For streaming Responses, success requires exactly one compaction
`response.output_item.done` with nonempty string `encrypted_content`, followed
by a successful `response.completed` whose output contains exactly one matching
compaction state. Text alone, an output-only terminal, duplicates, mismatches,
incomplete/error events, malformed events and premature EOF fail validation.
The encrypted state is opaque: the gateway cannot verify its semantic contents.

The native streaming path forwards progress and text incrementally. It holds
compaction added events, output-item done events (also cached by Codex's shared
client), and completion until validation succeeds. Holding all done items avoids
polluting that cache with ordinary items from a failed attempt. Buffers use the
existing precommit event/byte limits, and each event has the same byte cap.
No failed compaction state or successful terminal is sent before failover.
Receiving text does not commit a compaction result.

The generic compatibility path uses the existing bounded Responses collector,
then replays the validated winner's created/output-done/completed events. This
path delays progress until validation; it does not retain the full SSE stream.
Native nonstream JSON and stream-only collection validate before returning a
result. The independent `/compact` JSON endpoint accepts its documented lack of
a `status` field and does not require an SSE done event.

Retries use the current AUTO_RETRY allowance, targeting constraints, routing and
timeouts. Compression attempts are serial, each provider is visited at most once,
and cancellation stops native streaming work. No extra probe is sent. Explicit
unsupported-capability HTTP/SSE errors map to `compaction_unsupported`; invalid
results to `compaction_invalid_output`; failed/interrupted streams to
`compaction_stream_failed`. Auth/quota/arbitrary HTTP errors retain their current
classification. Semantic compaction failures do not cool ordinary chat routes.
No permanent capability or channel configuration is changed automatically.

Each failed attempt remains a channel failure even if a later provider succeeds;
the final request succeeds only for a validated winner. Failure reasons use the
existing fact fields, with no new sensitive payload persistence.

Verification: `cargo test --locked`, `cargo clippy --locked --all-targets -- -D warnings`,
`python3 tests/http/verify_compaction_retry.py target/debug/uni-api-front`, and
`python3 tests/http/verify_compaction_routing.py target/debug/uni-api-front`.
