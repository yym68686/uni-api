# Upstream channel balances

`GET /v1/channel-balances?provider=<configured-provider-name>` is a platform
endpoint. Only the first configured uni-api key or an `admin` / `admin-*` key
may call it. Ordinary uni-api keys cannot inspect balances, including their own.

The adapter uses the provider's existing upstream credential to call sub2api
`GET /v1/usage` with `Authorization: Bearer <upstream-key>`. Sub2api accepts an
ordinary upstream API key; a dashboard login or admin token is unnecessary.
Disabled users/keys and IP restrictions still apply. Sub2api skips balance and
quota admission for this read endpoint, so exhaustion does not by itself prevent
checking the remaining amount. Its auth middleware may update the key's last-used
timestamp; the query does not generate model traffic or spend model credits.

The URL is derived from an HTTPS provider base ending in `/v1`, `/v1/responses`,
`/v1/messages`, or `/v1/chat/completions`, preserving a deployment prefix. A bare
HTTPS origin also works. Redirects are blocked and the caller cannot supply a URL
or credential. Official OpenAI, Anthropic, DeepSeek and unsupported URL forms
are marked unsupported. Set provider `preferences.balance_query: false` to opt
out. No routing selection, limits, cooldowns or model statistics are modified.

Each result contains `keys` in configured credential order, with a numeric
`position` and `status`. Successful entries include `kind`, `amount`, `currency`,
`unlimited`, `windows`, `key_valid` and `checked_at`. Full keys, upstream error
bodies, usage logs and account identity are never returned. A provider with more
than eight distinct keys reports `omitted_keys` explicitly. Duplicate credentials
are queried once; amounts are never added together because keys or providers may
share a wallet.

- `wallet`: account wallet balance.
- `key_quota`: remaining total quota of that particular upstream key.
- `key_rate_limits`: remaining amounts for individual 5h/1d/7d windows.
- `subscription`: subscription remaining allowance; `unlimited` represents the
  sub2api `-1` sentinel. Missing data stays null and is not displayed as zero.

Queries use an isolated HTTP client, at most three global upstream requests at
once, six-second request timeouts and a 256 KiB response cap. Results and failures
are cached for five minutes per URL/credential/proxy; concurrent requests for the
same tuple are coalesced. The cache is bounded to 512 entries. Optional sub2api
usage queries are restricted to one day. Cache timestamps are included; data is
volatile and requires no database.

The frontend fetches balances independently after rendering channel metrics,
queries each distinct displayed provider once per refresh and shows unsupported,
denied, timeout and error states separately. Hovering an amount shows its query
time. The same provider's amount repeats on its model rows for convenience and
does not represent a separate balance for each model.
