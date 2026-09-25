# Native Responses stage observations

Native streaming Responses attempts add an optional `transport_timing` object
to immutable attempt facts and existing upstream attempt logs. Existing latency
fields retain their meaning; no routing, payload, timeout, retries, headers or
stream flushing policy changes. Other protocol paths report null rather than
inventing measurements. Old facts also lack the new fields.

All durations use the same monotonic attempt HTTP send clock:

- `headers_received_ms`: reqwest send resolves with HTTP response headers. This
  includes connection selection/establishment, request upload and server waiting;
  it does not independently identify which part was slow.
- `first_upstream_chunk_ms`: first nonempty body chunk observed by this process.
- `public_stream_ready_ms`: preflight/hedge selection finished and public body is
  being constructed. This is not a network write or client receipt timestamp.
  Opaque passthrough can reach this stage before its first upstream body chunk.
- `first_wire_prepared_ms`: first nonempty downstream bytes prepared by the
  existing wire accounting hook. It is not a successful flush measurement.
- `preflight_decode_ms`: accumulated synchronous SSE frame decoding before
  commit; does not include network waiting or all semantic processing.
- `preflight_read_wait_ms` / `preflight_read_calls`: accumulated time and number
  of awaited reads before commit, including timer wakeups. Waiting includes task
  scheduling and the upstream stream, so it is not a pure network-latency metric.
- `preflight_process_ms`: elapsed synchronous frame processing before commit,
  including semantic checks and buffering. It is wall time, not CPU profiling.
- `error_body_read_ms`: elapsed bounded error-body read for non-success HTTP
  responses. These attempts retain the already observed header time; unknown
  body-first-byte and public output stages remain null.

Missing observations are null. `network_write_measured=false` explicitly marks
the boundary. Join by request_id+attempt_id and compare existing response_created,
first_output and first_text. Record model, body size, input/cache sizes and error
population before comparing different deployment sites. A difference between
unmatched request distributions is not causal evidence of a gateway bug.
SSE read errors, EOF, deadline outcomes and semantic preflight retries preserve
available stage data. Failures before response headers and decoder exceptions
can still lack it. Never infer that a missing measurement is zero.
