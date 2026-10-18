# Claude thinking and effort conversion

For Chat/Responses requests converted to Anthropic Messages, the verified actual
upstream models claude-sonnet-5 and claude-haiku-5-5 map reasoning intent to adaptive thinking.
For example, reasoning_effort=high becomes thinking={type:adaptive} with
output_config={effort:high}.

The capability match is exact and uses the mapped target model (including a
Messages provider's outgoing model override), not the client's alias. Older or
unknown targets retain the existing manual budget mapping; model-name prefixes
are not capability evidence. No thinking is invented without thinking/effort intent.

- Explicit thinking wins, including disabled and invalid values that the upstream
  must validate. Valid legacy Haiku 5.5 manual settings are normalized as described below. Disabled mode preserves sampling parameters
  and is never re-enabled by reasoning effort.
- The entire output_config survives conversion. Existing effort wins; otherwise
  reasoning_effort, then reasoning.effort, supplies it for adaptive mode.
- Adaptive mode preserves max_tokens / Responses max_output_tokens and does not
  synthesize a budget or enlarge the output limit. Manual enabled mode keeps its
  existing budget handling.
- Unknown effort values are not silently downgraded or guessed. Explicit invalid
  values remain available to upstream validation.
- Forced tool selection and tool history are not disabled or rewritten to make a
  request succeed. Native Messages retains its explicit payload and signatures.
- Responses extensions are forwarded only into Anthropic adapters, not unrelated
  provider protocols. Channel overrides apply before the final Haiku wire normalization.
- HTTP failure classification, retry policy and committed-output behavior are unchanged.

Verification checks actual outbound payloads for streaming and nonstreaming Chat
and Responses, mapped/overridden target models, native Messages, explicit field
precedence, legacy budgets, output limits, and ordinary validation 400 responses.

## Haiku 5.5 legacy native Messages compatibility

At the Anthropic wire boundary, the exact actual model claude-haiku-5-5
normalizes thinking.type=enabled with an integer budget_tokens >= 1024 to
thinking.type=adaptive, removing budget_tokens. This handles native Messages as
well as converted Chat/Responses and model aliases/overrides. It also runs before
manual-budget expansion during Chat conversion, so the caller's output limit is
not enlarged as a side effect.

No effort is guessed from a token budget. Existing output_config, including its
effort and format, stays intact. Explicit disabled/adaptive modes, unknown models,
older Haiku models and malformed legacy budgets remain available to ordinary
upstream validation. Messages, signed thinking history and tool records are not
changed by this normalization.

Haiku conversion no longer injects sampling parameters. Explicit temperature,
top_p or top_k values remain in the request for upstream validation rather than
being silently replaced or removed. Status classification and retry behavior are
unchanged.

Regression tests cover all three endpoints, streaming/nonstreaming, model
overrides in both directions, disabled/no-thinking cases, bad budgets, sampling
validation, effort precedence and signed native tool history.
