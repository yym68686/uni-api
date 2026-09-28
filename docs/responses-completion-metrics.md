# Responses completion statistics

For native streaming `/v1/responses`, a successful attempt requires the upstream
`response.completed` terminal SSE event. HTTP 200, text deltas and `[DONE]` are
not completion evidence. `response.incomplete` (including `max_output_tokens`),
EOF without completion, protocol errors and transport failures count as failed
attempts, even if the public stream already began with HTTP 200 or partial text.
A completed tool call does not need assistant text to count as successful.

The gateway applies this definition to live channel metrics, stored channel and
request statistics, and immutable S3 facts. Request/attempt facts add nullable
`terminal_kind` and `response_completed` fields. `response_completed` is a boolean
only for observed native streaming `/v1/responses`; non-streaming requests and
other protocols leave it null. HTTP status and semantic completion are distinct.

This changes measurement, not response bytes, route configuration, retries or
cooldown policy. In particular, an incomplete terminal response remains terminal;
the gateway does not silently replay a paid request after streaming began.

Older exporters collapsed incomplete responses into `outcome=success`, losing
termination evidence. Such historical success facts cannot safely be relabeled
by looking at token counts or missing first-output latency. Consumers retain the
legacy outcome when completion evidence is absent, and identify this limitation
in their metric explanation. Explicit historical `outcome=incomplete` can be
counted as failed without guessing.

Run `python3 tests/http/verify_responses_success.py target/debug/uni-api-front`
for isolated real HTTP/SSE checks, including incomplete after blank/real text,
EOF, `[DONE]`, normal completion and tool-only completion across provider modes.
