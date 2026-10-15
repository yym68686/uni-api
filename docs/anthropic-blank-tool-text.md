# Blank assistant text in Claude tool turns

Some Bedrock-backed upstreams reject text blocks containing only whitespace.
An observed `/v1/messages` request included a signed thinking block, a text
block containing one space, and two tool calls in the same assistant turn.

Before sending an Anthropic-format request, uni-api removes plain blank text
blocks only from assistant messages containing a structurally valid tool call.
Other blocks, thinking signatures, tool arguments/results, meaningful whitespace,
annotations and cache-control metadata remain unchanged. The caller's original
request remains unchanged; each attempt prepares its own outgoing body.
This also applies after OpenAI-to-Claude conversion and before AWS signing.

Empty standalone messages and other validation failures retain their existing
400 behavior. This compatibility fix does not classify every Bedrock
ValidationException as a transient provider failure.

Authorized replays of both the original request and the version with the single
blank block removed completed with HTTP 200 on the same configured channel.
The historical 400 was not reproduced, so these replays alone do not prove a
causal fix. Offline HTTP tests use an upstream that consistently rejects the
original blank block and accepts the normalized request on its first attempt.
