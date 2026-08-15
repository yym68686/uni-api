FROM lukemathwalker/cargo-chef:0.1.71-rust-1.84-bullseye@sha256:e372d5aae4166598a5e4ce08c0eb75af0e7ec909d7e19188ebf669d79fa62355 AS rust-chef

FROM python:3.11-bullseye AS native-dependencies
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}
COPY --from=rust-chef /usr/local/cargo /usr/local/cargo
COPY --from=rust-chef /usr/local/rustup /usr/local/rustup
WORKDIR /workspace/rust/uni-api-native
COPY rust/uni-api-native/Cargo.toml rust/uni-api-native/Cargo.lock ./
COPY rust/uni-api-native/.cargo ./.cargo
RUN mkdir -p src/bin/uni-api-front && \
    printf 'fn main() {}\n' > src/bin/uni-api-front/main.rs && \
    printf '' > src/lib.rs && \
    cargo chef prepare --recipe-path recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json

FROM native-dependencies AS native-builder
WORKDIR /workspace
COPY README.md pyproject.toml ./
COPY uni_api/api/codex_models_pro_0_144_0.json ./uni_api/api/codex_models_pro_0_144_0.json
COPY static ./static
WORKDIR /workspace/rust/uni-api-native
COPY rust/uni-api-native ./
RUN cargo build --release --locked && \
    cp target/release/lib_uni_api_native.so /tmp/lib_uni_api_native.so && \
    cp target/release/uni-api-front /tmp/uni-api-front

FROM python:3.11 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

FROM python:3.11-slim-bullseye AS legacy-runtime
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT} \
    UNI_API_RUNTIME=hybrid
EXPOSE 8000
WORKDIR /home
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
COPY --from=native-builder /tmp/lib_uni_api_native.so /home/uni_api/_uni_api_native.so
COPY --from=native-builder /tmp/uni-api-front /usr/local/bin/uni-api-front
ENTRYPOINT ["/usr/local/bin/uni-api-front"]

FROM debian:bullseye-slim AS rust-runtime
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT} \
    UNI_API_RUNTIME=rust \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 8000
WORKDIR /home
COPY --from=native-builder /tmp/uni-api-front /usr/local/bin/uni-api-front
ENTRYPOINT ["/usr/local/bin/uni-api-front"]
