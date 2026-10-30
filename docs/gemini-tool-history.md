# Gemini tool history across OpenAI-compatible endpoints

Applies to Chat/Responses requests converted to Gemini/Vertex generateContent.
It does not change native Gemini wire payloads, explicit wire overrides, GPT or
Claude adapters, status-code retry matching, or provider/key cooldown policy.

## Identity and thought signatures

The adapter preserves a Gemini functionCall's original ID and its own Part's
thoughtSignature together, including streaming responses where both are present.
The returned OpenAI call ID is an opaque, versioned `call_g2_….<ordinal>` envelope;
clients must return it unchanged as the tool result's call ID. A native ID without
a signature is also enveloped to avoid mistaking native IDs for legacy signature
encodings. Existing `call_<base64url(signature)>.<ordinal>` IDs remain accepted.

Explicit `thoughtSignature`, `thought_signature` and
`extra_content.google.thought_signature` on a Chat tool call or Responses
function_call are accepted. Conflicting or malformed metadata is rejected. The
signature is returned only to its original functionCall Part, never copied to
other parallel calls. No global tool-signature cache is required. An unsigned
history such as `call_1` remains unsigned: the gateway cannot reconstruct a real
signature that the client never returned, and does not fabricate one.

Streaming text is still forwarded incrementally. FunctionCall Parts are retained
until the Gemini terminal frame, bounded to 1 MiB/1024 Parts, so a later signature
for the same native ID cannot change an already emitted OpenAI ID. Same-ID object
fragments merge only nonconflicting fields; ambiguous/conflicting fragments fail
as protocol errors rather than being guessed or duplicated. Truncated upstream
streams do not emit fabricated completed tool calls.

## Parallel turns

Adjacent Responses function_call items form one Gemini model content, just as
multiple tool_calls on one Chat assistant message do. The next batch of outputs
is matched by call_id and emitted in original call order as one user content
containing multiple functionResponse Parts, with matching native IDs. Names alone
are not used for association, so same-name parallel calls and reverse result
arrival are safe. History is never regrouped across ordinary user messages or
across completed tool rounds.

Duplicate IDs within a batch, missing results, duplicate outputs, orphan outputs,
invalid function arguments and interruption by another message are rejected
before sending a malformed Gemini request. Local 400 errors expose
`code=gemini_tool_history`, `type=invalid_request_error` and a `param` field path;
they do not echo arguments, results, signatures or caller IDs.

## Parts and errors

Existing cleanup of exactly empty assistant tool text (`""`, null, pure empty
text Parts) remains. Whitespace, nonempty text, parameters and results are not
rewritten. Text-only requests never manufacture inlineData. Explicit image data
URLs must contain an image MIME type and nonempty valid base64 data; failures
identify the message/content path with `code=gemini_part`. Native Gemini and
explicit contents overrides are not normalized by this rule.

Upstream error message/type/code/param are preserved when provided, including
Gemini SSE failures after text output; a generic message is not assigned an
invented parameter or upstream code. Already-started streams do not retry.

All regressions in `verify_gemini_tool_history.py` use isolated loopback fixtures.
A mock HTTP200 is not evidence that a production provider accepted a request.

References: [Gemini Part/FunctionCall/FunctionResponse schema](https://ai.google.dev/api/generate-content#FunctionCall),
[OpenAI parallel function calling](https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling).
