# Claude thinking and effort conversion

For Chat/Responses requests converted to Anthropic Messages, the verified actual
upstream model claude-sonnet-5 now maps reasoning intent to adaptive thinking.
For example, reasoning_effort=high becomes thinking={type:adaptive} with
output_config={effort:high}.

The capability match is exact and uses the mapped target model (including a
Messages provider's outgoing model override), not the client's alias. Older or
unknown targets retain the existing manual budget mapping; model-name prefixes
are not capability evidence. No thinking is invented without thinking/effort intent.

- Explicit thinking wins, including disabled, manual settings and invalid values
  that the upstream must validate. Disabled mode preserves sampling parameters
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
  provider protocols. Existing channel overrides continue to apply last.
- HTTP failure classification, retry policy and committed-output behavior are unchanged.

Verification checks actual outbound payloads for streaming and nonstreaming Chat
and Responses, mapped/overridden target models, native Messages, explicit field
precedence, legacy budgets, output limits, and ordinary validation 400 responses.
