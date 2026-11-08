# Passive routing for explicit Claude cache prefixes

Native `/v1/messages` requests can reuse an explicit cache prefix while changing
the question after its breakpoint. Some suppliers instead report a large cache
write on each new question, although exact repeats read cache successfully.
The gateway now learns this observable cost pattern from completed real requests
and can select another eligible channel for later questions with that prefix.
This policy does not diagnose a supplier's internal implementation.

## Eligibility and evidence

- Inspect the final outgoing JSON without changing its body, cache markers or TTL.
- Initially support exactly one explicit `ephemeral` **text** breakpoint in system
  or message content, with default/`5m` or `1h` TTL. Multiple breakpoints, automatic
  caching, translated requests and unsupported shapes remain unclassified.
- Require at least 8 KiB of prefix, at most 2 KiB of suffix (also no more than a
  quarter of the prefix), and both reported cache read/write counts totaling at
  least 1,024 tokens. These are conservative eligibility bounds, not tokenization.
- Match caller key, channel, actual upstream credential, URL, model, provider
  preferences, request parameters, Anthropic version/beta headers, and the exact
  structured prefix. Changed keys, settings or prefixes form separate scopes.
- The baseline is the smallest observed read + write total in the current scope.
  After warmup, two sequential **different new suffixes** must each read less
  than half and write at least half of that baseline. Overlapping, unsuccessful,
  incomplete and missing-usage responses do not establish a rewrite.
- Only subsequent, previously unseen request variants are avoided. Still-warm
  exact repeats remain eligible. A repeat hit does not prove changed-suffix reuse.
  A new variant reading at least 80% and writing at most 20% clears the evidence.

## Availability and bounds

Skipping a channel sends no upstream request and consumes no dispatch/retry
budget. The successful request that supplies evidence is returned normally;
the gateway never replays it for cache optimization. Other channel cooldowns,
error policies, retry budgets and billing remain authoritative. If selection
exhausts the candidates without using the dispatch budget, the first deferred
channel is available as a last resort. A sole channel is served directly.

Evidence expires with the requested TTL, measured from request start. Rewrites
do not extend it; observed healthy reads can refresh it. State is process-local,
resets at restart, and is bounded to 2,048 scopes / 16 recently observed variants
per scope. A full variant history also leaves routing unchanged, since an evicted
exact repeat cannot safely be identified as a new suffix. Lock contention or
unclassifiable data leaves routing unchanged.
No probes, cross-caller learning or supplier-wide bans are involved.

Set `UNI_API_EXPLICIT_CACHE_GUARD=false` to disable the feature for a process, or
set a provider's `preferences.explicit_cache_guard` to `false` through the existing
validated channel settings API. Default is enabled. Disabling is immediate for
new selections and does not cancel requests already sent.

## Request trace

Trace facts record `skip_reason=explicit_cache_prefix_rewrite`, the action,
breakpoint path, requested TTL, rewrite count, token baseline, last evidence
request ID and remaining validity. Observations include actual read/write usage;
fallback is labeled `availability_fallback`. Prefixes appear only as HMACs with
a random process secret; neither prompts nor credentials enter these fields.
These diagnostic trace facts do not create billable attempts or usage samples.

The console's log detail translates the reason and exposes the evidence in its
expandable stage details. Historical requests cannot acquire missing evidence.

Verification: `cargo test --locked`, `cargo clippy --locked --all-targets -- -D
warnings`, and the loopback HTTP regression
`python3 tests/http/verify_explicit_cache_routing.py target/debug/uni-api-front`.
The shared HTTP runner includes this regression automatically.
