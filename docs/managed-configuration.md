# Console-managed configuration

Managed configuration is opt-in. Deploy a compatible `uni-api-web` analytics
backend first. Use two variables to select the backend and authenticate:

| Name | Value |
| --- | --- |
| `UNI_API_MANAGED_CONFIG_URL` | Private analytics backend URL ending in `/v1/runtime-config/<source-id>` |
| `UNI_API_MANAGED_CONFIG_ADMIN_KEY` | The source's existing administrator key; provision through the deployment's secret mechanism |
| `UNI_API_MANAGED_ALLOW_LOCAL_SEED` | Optional; defaults to `true` for a source that has never been configured. Set `false` for an already migrated deployment. |

Use HTTPS if the control-plane endpoint crosses an untrusted network. The
loader does not follow redirects or use proxy environment variables. The key
is not inserted into the model key catalogue and does not grant additional
gateway authorization. A complete bundle must contain the currently
configured administrator as an administrator, in addition to the existing
business keys and providers. When the managed URL is unset, standalone startup
and existing `UNI_API_CONTROL_RESTORE_*` behavior are unchanged.

While waiting for a validated initial bundle the process is alive, but the
business port is closed. This works with TCP readiness checks. The gateway
polls for versions, validates the base and overlay together, and switches
them together. Once initialized, a network/validation/revision error leaves
the last good version serving; an empty replacement will wait instead of
using stale YAML. Cold boot does not depend on the old PVC. Other workload
data, including spool files and local databases, are not configuration
and are not copied.

Legacy management URLs keep their authorization and, in managed mode,
commit configuration to the console before installing the change locally.
`POST /v1/add_credits` requires an `Idempotency-Key` in managed mode; use the
same value for retries of the same credit operation. Credit changes are
durably deduplicated in PostgreSQL.

For migration, first compare the online export against the expected base and
rules, then validate a no-YAML replacement using a real business probe.
Confirm the replacement's `managed_config_revision` on
`/v1/channel-controls` using authorized credentials. Do not remove the old
storage or change routing based only on the process starting.
