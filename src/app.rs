use crate::config::source::RuntimeConfigPublisher;
use crate::runtime::context::{env_bool, AppState};
use axum::routing::any;
use axum::Router;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let public_port = env_u16("PORT", 8000)?;
    let database_disabled = env_bool("DISABLE_DATABASE", false);
    let publisher = RuntimeConfigPublisher::discover(database_disabled)?;
    publisher.publish().await?;
    publisher.start_watcher();
    let persistence = crate::storage::database::Persistence::initialize(database_disabled).await?;
    let state = AppState::new(String::new(), String::new(), false, persistence, publisher)?;
    let _ = state.runtime.refresh().await;
    state.runtime.start_watcher();

    state.runtime.restore_controls_on_start().await?;

    let app = Router::new()
        .fallback(any(crate::api::handler::handler))
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
