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


### Raw committed stream checkpoints

`raw_stream` is null for paths that have not entered the raw committed loop.
For that loop it contains an entry timestamp, fixed-size totals, and one-shot
snapshots at the first observed `response.created`, substantive output and text.
Snapshots are taken after parsing the received chunk that contains the event;
coalesced frames in that chunk can be included. Events seen in preflight do not
create misleading raw-stage checkpoints. Missing and legacy checkpoints are null.

- `upstream_read_wait_ms` and `upstream_read_calls` count awaited body reads.
  This includes executor scheduling and any upstream/network waiting.
- `process_ms` measures synchronous SSE decoding and frame inspection wall time.
  It is not a CPU profile. `max_pending_frame_bytes` describes buffered partial
  events without storing their contents.
- `output_send_ms` and `output_calls` measure the existing send routine, including
  idempotency capture and waiting for the bounded output queue. They do not prove
  a network flush or receipt by the client.
- Frame/comment counts and the last nonempty chunk timestamp help distinguish
  early keepalives and incomplete events from a complete semantic response.

Only scalar metadata and at most three fixed-size milestone snapshots are kept;
there are no per-frame logs or payload samples. No forwarding, frame parsing,
terminal filtering, queue size, timeout or retry decisions change. A latency
partition localizes a wait boundary; it does not by itself prove a software bug.


### Correlating the physical outbound socket

`upstream_http_version` reports the response protocol. Optional `connection`
contains only `local_addr` and `remote_addr` from the HTTP client's existing
transport metadata. The local address belongs to the gateway's outbound socket,
not the caller. Neither URI, authorization, request/response headers nor body
contents are included. A proxy can make the physical peer different from the
ultimate origin; the tuple must not be described as a verified provider host.
A connector without metadata reports null, including failures before headers.

The tuple is an observation after headers, not a connect or write timestamp.
Join it with bounded, read-only socket counters in the same network namespace
and the attempt interval. HTTP/1 keepalive can reuse the tuple across sequential
requests. HTTP/2 can multiplex attempts; tuple counters alone then cannot be
attributed to one request. No DNS, TLS, proxy, protocol selection, pooling or
retry behavior changes to obtain this metadata.
