mod channel_balances;
mod channel_catalog;
mod channel_metrics;
mod codex_oauth;
mod config;
mod cors;
mod generic_api;
mod hedging;
mod idempotency;
mod native_api;
mod persistence;
mod provider_stream;
mod proxy;
mod request_decompression;
mod request_spool;
mod request_timing;
mod resources;
mod responses;
mod responses_item_ids;
mod responses_native;
mod telemetry;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use axum::routing::any;
use axum::Router;

use config::RuntimeConfigPublisher;
use proxy::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let public_port = env_u16("PORT", 8000)?;
    let database_disabled = env_bool("DISABLE_DATABASE", false);
    let publisher = RuntimeConfigPublisher::discover(database_disabled)?;
    publisher.publish().await?;
    publisher.start_watcher();
    let persistence = persistence::Persistence::initialize(database_disabled).await?;
    let state = AppState::new(String::new(), String::new(), false, persistence, publisher)?;
    let _ = state.native_responses_config.refresh().await;
    state.native_responses_config.start_watcher();

    let app = Router::new()
        .fallback(any(proxy::handler))
        .with_state(state);
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), public_port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("uni-api Rust runtime listening on {address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn env_u16(name: &str, default: u16) -> Result<u16, String> {
    let raw = std::env::var(name).unwrap_or_else(|_| default.to_string());
    raw.parse::<u16>()
        .map_err(|_| format!("{name} must be a valid TCP port"))
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
