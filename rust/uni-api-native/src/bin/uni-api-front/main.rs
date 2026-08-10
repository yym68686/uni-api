mod proxy;
mod responses;

use std::future::IntoFuture;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::routing::any;
use axum::Router;
use sha2::{Digest, Sha256};
use tokio::process::{Child, Command};

use proxy::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let public_port = env_u16("PORT", 8000)?;
    let backend_port = env_u16("UNI_API_PYTHON_PORT", 18001)?;
    if public_port == backend_port {
        return Err("PORT and UNI_API_PYTHON_PORT must differ".into());
    }
    let control_token = control_token();
    let backend_origin = format!("http://127.0.0.1:{backend_port}");
    let state = AppState::new(backend_origin.clone(), control_token.clone())?;
    let mut child = spawn_python(backend_port, &control_token)?;
    wait_for_backend(&state, &mut child).await?;

    let app = Router::new()
        .fallback(any(proxy::handler))
        .with_state(state);
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), public_port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("uni-api Rust frontend listening on {address}; Python control plane on 127.0.0.1:{backend_port}");

    let server = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => {
            result?;
        }
        status = child.wait() => {
            return Err(format!("Python control plane exited unexpectedly: {}", status?).into());
        }
    }

    if child.id().is_some() {
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    }
    Ok(())
}

fn spawn_python(port: u16, token: &str) -> Result<Child, std::io::Error> {
    let mut command = Command::new(
        std::env::var("UNI_API_PYTHON_EXECUTABLE").unwrap_or_else(|_| "python".to_owned()),
    );
    let python_main = std::env::var("UNI_API_PYTHON_MAIN").unwrap_or_else(|_| "main.py".to_owned());
    let responses_data_plane =
        std::env::var("UNI_API_RUST_RESPONSES_DATA_PLANE").unwrap_or_else(|_| "1".to_owned());
    command
        .arg(python_main)
        .args(std::env::args_os().skip(1))
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .env("UNI_API_RUST_FRONTEND_CHILD", "1")
        .env("UNI_API_RUST_RESPONSES_DATA_PLANE", responses_data_plane)
        .env("UNI_API_RUST_CONTROL_TOKEN", token)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command.spawn()
}

async fn wait_for_backend(state: &AppState, child: &mut Child) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("failed to inspect Python child: {error}"))?
        {
            return Err(format!(
                "Python control plane exited during startup: {status}"
            ));
        }
        match state
            .backend_client
            .get(state.internal_url("/healthz"))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            _ if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            _ => {
                return Err("Python control plane did not become healthy within 90 seconds".into())
            }
        }
    }
}

fn env_u16(name: &str, default: u16) -> Result<u16, String> {
    let raw = std::env::var(name).unwrap_or_else(|_| default.to_string());
    raw.parse::<u16>()
        .map_err(|_| format!("{name} must be a valid TCP port"))
}

fn control_token() -> String {
    let mut hasher = Sha256::new();
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    hasher.update(format!("{:p}", &hasher).as_bytes());
    format!("{:x}", hasher.finalize())
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
