# Responses heartbeat context repair

Some Codex histories contain scheduled notifications serialized as
`function_call_output` items without a `call_id`. When an upstream rejects the
history with the exact error below, native `/v1/responses` can repair these
notifications and resend to the same channel once:

```json
{"error":{"code":"function_call_output_not_found","message":"The tool output does not match a previous tool call. Resend the tool-call context.","param":"input","type":"invalid_request_error"}}
```

The initial request is unchanged. Repair requires HTTP 400, this structured
error (including bounded JSON wrappers in `error.message`), and no committed
business output. The repair applies only to items with all these properties:

- `type` is `function_call_output`.
- `name` is `automation_update` and `namespace` is `codex_app`.
- `call_id` is absent, null, or an empty string.
- `output` is a string enclosed in `<heartbeat>...</heartbeat>`.

Each matching item becomes a user message with an `input_text` content item.
Its text is `Historical scheduled heartbeat context:\n` followed by the entire
original output. Actual tool calls and outputs, reasoning, and other request
fields stay intact. The number of notifications is not hard-coded to 16.

The resend retains the channel URL, upstream key, model mapping, proxy, and
request options. It gets a separate attempt ID, dispatch measurement, and
upstream outcome. There is at most one historical-context repair resend per
incoming request, shared with [empty tool-name repair](responses-empty-name-repair.md)
and [missing reasoning-item repair](responses-missing-item-repair.md),
including when hedging is enabled. After that attempt fails, existing failure
classification applies. Its modified body is local to that attempt; channel
configuration and the bodies sent to other channels are unchanged.

`AUTO_RETRY=false`, administrator-targeted diagnostics, `/v1/responses/compact`,
other error bodies, and failures after business output do not activate repair.
In particular, a real tool output with a nonempty `call_id` is never converted.

The structured `responses_heartbeat_repair` log event links the original and
repair attempt IDs and records `heartbeat_items_converted`. It contains no
request text or credentials. Original upstream failures remain visible as 400;
the final request summary records the actual result after repair.

Offline HTTP coverage lives in `tests/http/verify_heartbeat_retry.py`, including
same-key preservation with rotating keys, one-resend limits, ordinary failures,
streaming and nonstreaming requests, and a hedging race.
