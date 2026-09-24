# Gateway tests

Run Rust unit tests from the repository root with `cargo test --locked`.
The tests remain colocated with their implementation under `src/`.

Build the gateway with `cargo build --locked`, then run all isolated HTTP
regressions with `python3 scripts/verify-http.py`. Each `http/verify_*.py`
accepts the gateway binary as its first argument, launches local mock endpoints
and uses temporary configuration/storage. No production credentials are required.

- `http/`: executable black-box regression suites.
- `fixtures/`: stable protocol, config and SSE fixtures.
- `fixtures/manual/`: historical manual examples, not automatically executed.
- `support/mock_server.go`: manually started mock server.

The HTTP runner discovers all regression suites so a new `verify_*.py` is also
included in CI. Shared Python helpers stay beside the suites to preserve imports.
