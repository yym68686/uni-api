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

## Prompt compaction for other providers

The gateway offers the same `/v1/responses` compaction trigger and
`/v1/responses/compact` endpoint on converted Chat, Claude, Gemini and other
generation routes. It asks the selected provider to summarize a text transcript
without tools. It uses native compaction by default on Responses routes whose
mapped upstream model is GPT, Codex, ChatGPT, o1, o3 or o4, and prompt compaction
on other routes. For a Responses channel, `preferences.compaction_mode` can be
`auto`, `native`, `prompt`, or a per-model map such as:

```yaml
preferences:
  compaction_mode:
    default: auto
    models:
      other-native-model: native
      gpt-compatible-without-compaction: prompt
```

Public model names take precedence over mapped upstream names in the map.
Non-Responses channels always use prompt compaction. Mode is picked per routed
attempt; native compaction failures do not switch modes on that same channel.

A successful summary is returned in a `compaction` item with
`encrypted_content: "compact-v1:<base64url>"`. The field name is for protocol
compatibility: this is **not encryption or an authenticity check**. Originating
clients must treat it as conversation data and send it back as an input item
along with subsequent input. The gateway restores valid checkpoints in place as
ordinary user messages, never as system or developer instructions. No key,
database, or cache is needed, including across restarts. `previous_response_id`
or `conversation` cannot substitute for the full input on this path. Native
opaque checkpoints cannot be decoded by the prompt route. Older gateway builds
do not recognize the new format, so a rollback to one of them can require
resending the original history for sessions that already hold a checkpoint.

Summaries are bounded to 32 KiB of UTF-8. Attachment content is not sent to the
summary model; text transcripts place a marker where attachments were present.
Reported usage is for the upstream summary generation, when available, not an
estimate of the size of the compressed history.

Verification: `cargo test --locked`, `cargo clippy --locked --all-targets -- -D warnings`,
`python3 tests/http/verify_compaction_retry.py target/debug/uni-api-front`, and
`python3 tests/http/verify_compaction_routing.py target/debug/uni-api-front`, plus
`python3 tests/http/verify_stateless_compaction.py target/debug/uni-api-front`
for cross-provider generation, error handling and restart replay.
