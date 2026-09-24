# Runtime observability

The previous Python worker document is retained in [the archive](archive/worker-observability.md).
It describes a historical implementation, including Uvicorn and Python stack sampling.

The Rust implementation is organized under `src/observability/` and `src/storage/`.
Runtime resource and idempotency diagnostics are assembled in `src/runtime/context.rs`
and exposed by `src/api/platform.rs`. See [architecture](architecture.md).
