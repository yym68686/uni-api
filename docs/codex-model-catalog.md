# Key-scoped Codex model catalog

GET /v1/codex/models and GET /v1/models?client_version=... return Codex-compatible model cards for the
authenticated API key's configured conversational routes. The query parameter's
presence selects this format; its value is ignored. An empty value, old version,
current version, or arbitrary future version returns exactly the same body and
ETag for the same key and configuration.

The model set follows the existing route graph, including provider/model rules,
global model rules, nested keys, and key-scoped temporary channels. Explicit
channel disablement and /v1/responses endpoint exclusions are respected. A model
stays listed when another permitted route remains. Temporary cooldowns, rate
windows and scheduling do not affect discovery. Listing does not make upstream
requests, reserve capacity, or mutate configuration.

Embedding, reranking, moderation, image generation (including gpt-image-2 and
gpt-image-2.5), speech/audio and video model families are omitted. The filter
checks both public aliases and their configured upstream model names. Providers
dedicated to the TypeSafe /v1/systemone protocol are also omitted.

For existing GPT/Codex entries, full embedded metadata is retained, including
hidden helper cards. gpt-6-astra.context_window remains 600000, with its existing
max_context_window of 872000.

Unrecognized GPT/Codex aliases use the gpt-5.6-sol compatibility card.
Other conversational names use generic cards with provider-neutral instructions
and compatibility context settings.

Gemini 4 Neon and Hydrogen offer low, medium, high and xhigh (default high).
Gemini 4 Carbon and Argon offer low, medium and high (default medium).
These fields are derived from the caller's effective upstream models and
protocols. Multiple routes for an alias advertise only levels supported by
all of them. The Responses and Chat adapters use the same capability
definitions to serialize requested levels instead of translating them to
legacy Gemini thinking budgets.

Ordinary GET /v1/models retains its existing OpenAI/TypeSafe response and
continues to include modality-specific models where authorized.

The rich response uses X-Uni-API-Models-Source: key-scoped-catalog, an ETag
computed over the final authorized body, Cache-Control: private, no-cache,
Vary: Authorization, X-Api-Key, and an explicit Content-Length. Conditional
requests are evaluated only after authentication and model filtering. The old
upstream snapshot ETag and fixed client-version headers are removed.

## Validation

Run cargo fmt --check, cargo clippy --all-targets -- -D warnings, cargo test
--locked and cargo build --locked with manifest Cargo.toml.
Then run python3 tests/http/verify_codex_models.py with the built uni-api-front
binary as its argument.

The HTTP fixture verifies per-key/nested grants, upstream aliases, all version
values, exact existing metadata, 600k Astra, new names, forbidden modalities,
scoped channel controls, ETag/auth isolation, unchanged ordinary catalogs and
absence of upstream calls. It is also part of the required Rust CI workflow.
