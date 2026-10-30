# Responses encrypted continuation fallback

For native `/v1/responses` and `/v1/responses/compact`, this exact upstream
JSON error is treated as a channel rejection of the request's encrypted history:

```json
{"error":{"type":"invalid_request_error","code":"encrypted_context_invalid","param":"input","message":"The upstream could not validate encrypted continuation context. Rebuild the conversation context in the client."}}
```

The outcome must be `http_error`, both status fields must be 400, and
`committed` must explicitly be false. The actual outbound `input` array must
contain a reasoning or compaction item with a nonblank string `encrypted_content`.
Any present invalid encrypted field on those item types prevents this fallback.
Compaction items without `encrypted_content` also prevent this new fallback.
Missing/null/different error fields, nested wrappers, quoted messages, ordinary
validation errors and failures after output are not new triggers.

Routing uses 502 and the existing retry budget and channel order. Raw attempt
and billing facts keep 400. With retries disabled, an explicit provider target,
or an exhausted budget, the result is 502 without an additional channel attempt.
This request-specific rejection does not cool the provider or its key globally.

The fallback never deletes, decodes, replaces or repairs encrypted history.
Each channel compiles the original request with its existing channel settings;
identical settings produce identical outbound bytes. No additional body changes
are introduced. The existing `responses_encrypted_content_fallback` event records
request/attempt IDs, provider/model, raw 400, routing 502 and error hash. It is
shared with the existing redacted encrypted-content rejection; the exact error
hash distinguishes the variants. No request content or credentials are logged.

Isolated tests cover GPT and Codex transports, streaming/nonstreaming/compact,
retry disable, exhaustion, explicit target, repeated requests without cooldown,
exact negative cases, committed output, raw billing and unchanged fallback body.
