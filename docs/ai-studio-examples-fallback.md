# AI Studio examples capability rejection

The relay can return HTTP 400 for an AI Studio schema feature it has not enabled.
Only this complete structured `error` is treated as a request-specific channel
compatibility failure:

- `type` is exactly `invalid_request_error`, `code` exactly `invalid_request`;
- `param` is absent, null or the empty string;
- the complete message is `AI Studio 请求参数无效: 编码 function declaration N: parameters: ***.PATH: AI Studio 协议能力尚无成功现场证据: JSON schema 字段 examples`;
- `N` is a canonical unsigned 32-bit decimal index, and `PATH` consists of one or
  more `properties.NAME` pairs with ASCII identifier names, for example
  `properties.duration_min` or `properties.options.properties.duration_min`.

The raw upstream attempt and billing remain 400; routing uses 502 and the existing
retry budget/channel selection. Disabled retry, an explicit provider target or an
exhausted budget returns 502. This request does not cool the provider/key for
unrelated traffic. After stream output is committed, existing no-retry behavior
continues to apply.

Request compilation, messages, tool schemas (including examples), sampling and
other settings are unchanged. This adds no field deletion or same-channel repair.
Plain strings, nested/quoted errors, alternate masks, other capability names,
ordinary parameter validation and other HTTP statuses keep their existing policy.
The `ai_studio_examples_fallback` event contains identifiers and status metadata,
never a request body, schema examples or credentials.
