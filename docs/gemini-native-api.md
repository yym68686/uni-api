# Gemini native API

The gateway accepts these POST routes with its existing gateway API keys:

- `/v1beta/models/{model}:generateContent`
- `/v1beta/models/{model}:streamGenerateContent`
- `/v1beta/models/{model}:countTokens`
- The same generation/count routes under `/v1/models/`.

The model in the URL is the configured public model alias. Authentication accepts
`x-goog-api-key`, `Authorization: Bearer ...`, `x-api-key`, or the `key` query
parameter. Query credentials are removed before request tracing and forwarding.
`GET /v1beta/models` and `GET /v1beta/models/{model}` expose the authorized catalog.
The existing `GET /v1/models` retains its OpenAI-compatible response format.

`streamGenerateContent?alt=sse` returns Gemini SSE without an OpenAI `[DONE]`
sentinel. Without `alt=sse` it returns a streamed JSON array. Both support the
existing idempotency and routing mechanisms. Errors use Google's `error` object.

## Provider conversion

Gemini and Vertex Gemini providers preserve the native request/response JSON,
including media, thought signatures, tools and native configuration. Model aliases,
provider credentials and administrator overrides are applied by the existing
provider preparation path. `countTokens` requires a Gemini or Vertex Gemini provider.

Chat, Responses and Messages providers accept Gemini text, inline image input,
system instructions, function declarations, function call/result history and
supported sampling controls. Their output is converted to Gemini candidates,
function calls, finish reasons and usage metadata. Fragmented function arguments
are assembled and validated before emitting a Gemini function call.

Native-only features such as cached content resources, hosted Google tools,
non-image media and exact thinking budgets require a native provider. Unsupported
fields cause a conversion error instead of silently disappearing. Responses and
Messages support one candidate per request. Responses conversion rejects `topK`,
`stopSequences` and `seed`; Messages conversion rejects `seed` and structured
output configuration. JSON Schema output is supported through Responses and Chat.

`POST /v1/responses` can also select a Gemini provider. Input, functions, reasoning,
structured output and streamed/non-streamed results use the existing conversion
pipeline. Cached and reasoning token details survive the return conversion.
Truncation and safety stops return an incomplete Responses result.

For converted Chat/Responses requests ending in ordinary assistant text, the
Gemini adapter preserves that text and appends a user turn containing `Continue.`.
This supports Gemini providers that reject a final `model` turn. It is an explicit
continuation instruction, not assistant-prefix completion. The adapter only adds
it when all function calls have a matching result in order; parallel call batches
are supported. Unresolved, duplicate or malformed tool records, non-text endings,
and unsupported Responses history items do not enable this normalization.
Native Gemini requests and explicit provider body overrides retain their existing
behavior. No status-code remapping or new retry policy is involved.

## Usage

Gemini `promptTokenCount` includes cached input. `candidatesTokenCount` excludes
`thoughtsTokenCount`; both count toward billed output. Conversion preserves those
semantics. Converted usage includes an additional `uniApiUsage` object containing
the upstream counters, including Anthropic cache creation/read splits and any
settlement receipt, for downstream gateways that need the original billing data.
Native Gemini responses are not extended or rewritten.

An error event or a stream ending without a terminal event is recorded as a
failure. HTTP 200 upstream error objects are not converted into successful output.

Run `python3 tests/http/verify_gemini_native.py target/debug/uni-api-front` for the
isolated protocol matrix, or `python3 scripts/verify-http.py` for all HTTP checks.
