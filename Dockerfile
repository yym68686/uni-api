FROM rust:1.84-bullseye AS rust-toolchain

FROM python:3.11-bullseye AS native-builder
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
WORKDIR /build
COPY rust/uni-api-native/Cargo.toml rust/uni-api-native/Cargo.lock ./
COPY rust/uni-api-native/.cargo ./.cargo
COPY rust/uni-api-native/src ./src
RUN cargo build --release --locked

FROM python:3.11 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

FROM python:3.11-slim-bullseye
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT}
EXPOSE 8000
WORKDIR /home
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
COPY --from=native-builder /build/target/release/lib_uni_api_native.so /home/uni_api/_uni_api_native.so
COPY --from=native-builder /build/target/release/uni-api-front /usr/local/bin/uni-api-front
ENTRYPOINT ["/usr/local/bin/uni-api-front"]
