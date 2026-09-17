# Dashboard channel selection

The platform dashboard accepts the first configured API key or a key with role
`admin` / `admin-*`. Ordinary keys can use model services and `/v1/models`, but
cannot read platform APIs, including their own key metadata. The first key's
catalog permission does not grant configuration mutation permissions.

`GET /v1/api-keys` returns configuration order, position, masked prefix, and a
stable opaque `key_id`. Full keys and raw rules (which may reference keys) are
never returned. Short keys are fully masked. Send the connection credential only
in `Authorization: Bearer ...` and the selected opaque ID in `api_key_id`.

These endpoints accept an optional `api_key_id`:

- `GET /v1/model-channels`
- `GET /v1/channel-metrics`
- `GET /v1/channel-metrics/timeseries`

Without a selection, the catalog includes all configured providers in provider
configuration order. With a selection, it includes only that key's channels,
expanded in configured rule order. Wildcards expand at their position; nested
key references keep their referenced rule order and model restriction; duplicate
provider/model pairs keep their first position. Plain model rules and literal
`<vendor/model>` rules expand across supporting providers. Models within a
provider wildcard are sorted for deterministic output. Endpoint exclusions still
apply, and cooled-down channels remain visible with their current status.

Responses include `order` (`provider_config` or `api_key_config`) and the selected
`api_key_id` (empty in default mode). A removed or unknown ID returns 404; it never
silently falls back to all channels. Credentials are validated before selection,
and ordinary keys receive 403 even when selecting their own ID.

Selection changes the displayed channel set and order. Metrics remain overall
provider/model/endpoint/stream statistics, not that key's private request usage;
metrics responses declare `statistics_scope=channel_all_requests`. Display order
is configuration order, not the currently rotated/random/weighted dispatch
sequence. Queries never select provider credentials, advance scheduling cursors,
consume rate limits, alter cooldowns, or call upstream models.

`endpoint=all` includes all endpoints, and `stream=all` combines streaming and
non-streaming attempts. Either filter can be used independently. The catalog
keeps one row per provider/model; aggregate rows use `endpoint="all"` and/or
`stream=null`. Explicit endpoint filtering still applies configuration endpoint
exclusions. Aggregate availability means the provider/model's general cooldown
and credential state, not a claim that every endpoint is supported.

Metrics merge raw minute buckets and latency histograms before calculating
success rates and p50/p95, both for totals and timeseries; quantiles are never
averaged. `filters` declares the applied scope and `available_endpoints` lists
observed endpoint names for selector discovery. Omitting these query parameters
preserves the existing API defaults (`/v1/responses`, `stream=true`). The console
explicitly requests `endpoint=all&stream=all` by default.

The generic request path now records the actual downstream streaming mode for
started, failed, completed and hedged attempt metrics. Previous releases labeled
all generic attempts as non-streaming. Historical process-memory metrics cannot
be relabeled reliably; after deployment, new samples start with correct labels.

Platform GET routes protected by this policy: `/v1/api-keys`, `/v1/model-channels`,
`/v1/channel-metrics`, `/v1/channel-metrics/timeseries`,
`/v1/observability/runtime`, `/v1/stats`, `/v1/token_usage`,
`/v1/channel_key_rankings`, `/v1/api_keys_states`, `/v1/api_config`, and
`/v1/generate-api-key`. Existing admin-only routes retain their additional checks.
`/healthz` remains the unauthenticated deployment health probe.

### Temporary imported channels

`POST /v1/temporary-channels` accepts a current `/v1/channel-controls` revision,
`api_key_id`, stable `provider` name (`sub2api-*`), `base_url` ending in
`/v1/responses`, upstream `api_key`, `models`, and a one-based `position`.
Only the first configured key or an administrator can call it. The mutation
atomically installs a process-local provider and model-specific priority rules.
Each selected model must have enough channels for the requested position.

An imported provider is restricted to the exact destination API key. Wildcards
and parent-key expansion do not grant other keys access; administrator-directed
diagnostics can still inspect/test it. Repeating an import with a fresh revision
updates the same provider rather than duplicating it. Secrets are never returned
in the controls view or audit event. The configuration file and snapshot artifact
are unchanged. Resetting the key/model control scope also removes imported models
in that scope; reset-all and process restart remove every temporary provider.
