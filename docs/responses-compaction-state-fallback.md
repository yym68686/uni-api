# Responses invalid compaction state fallback

An exact provider rejection may indicate that opaque Responses compaction state
is not accepted by that provider/account, while another configured provider can
complete the same unchanged request. For a matching uncommitted HTTP 400 only,
uni-api preserves the upstream attempt and billing status as 400 and uses the
existing request-scoped next-channel retry with routing status 502.

The match requires the complete structured `invalid_request_error` envelope:

- `code`: `invalid_compaction_state`
- `message`: `压缩状态无效或不属于当前调用方，请使用原调用凭据，或携带完整历史重试。`
- `param`: absent or null

An additional exact relay envelope is accepted with `type=upstream_error`,
`code` and `param` absent or null, and the complete message:

```text
The encrypted content for item cmp_<UUID> could not be verified. Reason: Encrypted content could not be decrypted or parsed.
```

Here `<UUID>` must have the 36-character hexadecimal UUID shape, including its
four hyphens. The complete ID must identify exactly one actual `compaction` input
item with a non-empty string `encrypted_content`. Duplicate IDs, a different item
type, quoted/nested errors, extra message text, concrete codes, or invalid/empty
state keep the ordinary error behavior. This rule does not enable same-channel
repair or broaden the existing Cursor reasoning-summary repair.

The original Responses input must contain at least one `compaction` item and
every such item must contain a non-empty string `encrypted_content`. The request
body is not repaired, decoded, deleted, or otherwise modified. Empty/malformed
state and ordinary validation, authentication, or unrelated errors retain their
normal status and retry policies. Retry-disabled or exhausted requests return
502; an already committed stream is never retried. This rejection does not add a
provider/key cooldown because the evidence is request-scoped, not proof that a
channel is generally unavailable.

`tests/http/verify_encrypted_content_fallback.py` exercises the local-only HTTP
contract for native and Codex Responses, streaming and non-streaming, exact
wire-body preservation, retry-disabled/exhausted/committed cases, and strict
negative matching. No production request is replayed by this test.
