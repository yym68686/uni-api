# All upstream HTTP 400 fallback

`preferences.unknown_upstream_400_retries` is the source-global 0–99 extra-channel
allowance. The persisted name remains unchanged for compatibility; it now
applies uniformly to upstream HTTP 400s rather than an error-message allowlist.
The default remains 0, so a release does not change production intent. An
attempt that succeeds ends fallback early; the configured count is a ceiling,
not a promise to dispatch that many extra requests. This budget is independent
of the caller key's `AUTO_RETRY` setting. The existing cap of 100 total attempts
and eligible channel availability still apply.

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
The gateway advertises the maximum in `all_upstream_400_retries_max` and sends
`X-Uni-All-Upstream-400-Retries-Max` at boot. A retained value above 3 is not
issued to a gateway that cannot honor it.

Fast and Ultrafast Responses routing is a separate policy enabled by default.
It tries verified channels for the requested tier first, then uses the ordinary
channel order without forwarding the tier. A structured rejection of
`service_tier` on an accelerated attempt can advance that route before any
substantive output, within the normal `AUTO_RETRY` budget. Other 400s retain the
policy above. Setting `preferences.service_tier_routing: false` retains legacy
tier forwarding. Targeted probes and ordinary requests keep their existing
behavior. Regression coverage uses loopback providers only.
