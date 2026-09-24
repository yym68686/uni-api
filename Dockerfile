FROM lukemathwalker/cargo-chef:0.1.71-rust-1.84-bullseye@sha256:e372d5aae4166598a5e4ce08c0eb75af0e7ec909d7e19188ebf669d79fa62355 AS chef
WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
RUN mkdir -p src && printf 'fn main() {}\n' > src/main.rs && touch src/lib.rs && cargo chef prepare --recipe-path recipe.json
FROM chef AS planner
RUN cargo chef cook --release --locked --recipe-path recipe.json
FROM planner AS builder
COPY Cargo.toml Cargo.lock ./
COPY README.md ./README.md
COPY assets ./assets
COPY src ./src
RUN cargo build --release --locked && cp target/release/uni-api-front /tmp/uni-api-front
FROM debian:bookworm-slim
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT} MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 UNI_API_RUNTIME=rust
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
EXPOSE 8000
WORKDIR /home
COPY --from=builder /tmp/uni-api-front /usr/local/bin/uni-api-front
ENTRYPOINT ["/usr/local/bin/uni-api-front"]
