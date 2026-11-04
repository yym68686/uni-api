# All upstream HTTP 400 fallback

`preferences.unknown_upstream_400_retries` is the source-global 0–3 extra-channel
allowance. The persisted name remains unchanged for compatibility; it now
applies uniformly to upstream HTTP 400s rather than an error-message allowlist.
The default remains 0, so a release does not change production intent. An
attempt that succeeds ends fallback early; 3 is a maximum, not a promise to
dispatch three more requests. This budget is independent of the caller key's
`AUTO_RETRY` setting, although existing total attempt and provider availability
limits still apply.

The policy uses upstream/protocol outcome status and the response commit
boundary, not error text, envelope fields, model, tool names, encrypted history,
or request-body features. It covers any upstream HTTP 400 and recognized SSE
semantic 400 that fails before substantive output is committed. The original
logical request is not repaired, stripped, truncated, or otherwise rewritten
for a retry. Each channel still uses its normal request compiler.

No retry occurs for local gateway validation, an explicitly targeted provider,
or a failure after substantive text/tool output has been committed; replaying
after output could duplicate work and cannot replace an already-started client
response. Existing opaque native streams retain their established commit
contract. Cooling or unavailable channels do not consume the extra allowance,
and a failed channel is not revisited under a second key. This policy does not
add a global provider/key cooldown or convert 400 into 502. Raw attempt and
billing status remain 400 (an SSE semantic 400 may have upstream HTTP 200); if
no eligible channel succeeds, the request ends with 400.

The console exposes this setting only when the gateway advertises
`all_upstream_400_retries`; it will not silently restore nonzero intent to an
older gateway. Existing persisted settings are preserved. The bootstrap header
`X-Uni-All-Upstream-400-Retries: 1` signals that capability during restore.

All former content-, provider-, and message-specific 400 retries and request
repairs are retired. Non-400 fallback policy and the independent 404 persisted
item recovery are unchanged. Regression coverage uses loopback providers only;
it does not replay customer traffic or call paid model providers.
