# Null text in Chat tool-call history

Chat assistant turns containing valid function calls can have `content: null`.
When translating Chat to Responses, function calls become separate Items. Do not
also create an assistant message with `content: []` for the null text placeholder.
The existing empty-string placeholder handling now also covers explicit null.

This applies only to assistant turns with a nonempty list of entirely valid
function calls: nonblank call IDs and names, and JSON-object argument strings.
Calls, argument strings, outputs, order, nonempty/whitespace text, and parameters
are unchanged. A missing content field already produces no text Item. Empty
arrays, other content shapes, standalone null messages, and malformed calls keep
their existing behavior; this is not a general request-validation bypass.

Native Responses input and explicit body overrides remain untouched. No error
status remapping, retry policy, provider selection, or production configuration
changes are introduced.
