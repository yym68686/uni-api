pub(crate) mod channels;
pub(crate) mod settings;

// Authenticated configuration has no fixed byte quota by default. Operators
// may set a positive budget; zero/unset leaves transport deadlines in control.
pub(crate) fn config_body_limit() -> usize {
    std::env::var("RUST_ADMIN_CONFIG_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(usize::MAX)
}
