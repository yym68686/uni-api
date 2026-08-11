use std::collections::HashSet;
use std::ffi::CString;
use std::fs;
use std::io;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceKind {
    Memory,
    FileDescriptor,
    EphemeralPort,
    Connection,
    LocalDisk,
    LocalInode,
}

impl ResourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory_headroom",
            Self::FileDescriptor => "file_descriptor_headroom",
            Self::EphemeralPort => "ephemeral_port_headroom",
            Self::Connection => "connection_headroom",
            Self::LocalDisk => "local_disk_headroom",
            Self::LocalInode => "local_inode_headroom",
        }
    }

    pub fn exhausted_status(self) -> StatusCode {
        match self {
            Self::LocalDisk | Self::LocalInode => StatusCode::INSUFFICIENT_STORAGE,
            _ => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResourceWait {
    pub waited: Duration,
}

#[derive(Clone, Copy, Debug)]
pub struct CapacityFailure {
    pub resource: ResourceKind,
    pub waited: Duration,
}

#[derive(Clone)]
pub struct ResourceGovernor {
    inner: Arc<Inner>,
}

struct Inner {
    cached: Mutex<CachedGlobalSample>,
    sample_interval: Duration,
    wait_timeout: Duration,
    memory_reserve_bytes: u64,
    memory_reserve_bps: u64,
    descriptor_reserve_bps: u64,
    port_reserve_bps: u64,
    disk_reserve_bps: u64,
    inode_reserve_bps: u64,
}

struct CachedGlobalSample {
    sampled_at: Instant,
    rejection: Option<ResourceKind>,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalCapacity {
    pub free_bytes: u64,
    pub writable_bytes: u64,
    pub free_inodes: u64,
    pub writable_inodes: u64,
}

impl ResourceGovernor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                cached: Mutex::new(CachedGlobalSample {
                    sampled_at: Instant::now()
                        .checked_sub(Duration::from_secs(1))
                        .unwrap_or_else(Instant::now),
                    rejection: None,
                }),
                sample_interval: Duration::from_millis(env_u64(
                    "RUST_RESOURCE_SAMPLE_INTERVAL_MS",
                    250,
                )),
                wait_timeout: Duration::from_secs_f64(env_f64(
                    "RUST_RESOURCE_WAIT_TIMEOUT_SECONDS",
                    120.0,
                )),
                memory_reserve_bytes: env_u64("RUST_MEMORY_RESERVE_BYTES", 64 * 1024 * 1024),
                memory_reserve_bps: env_basis_points("RUST_MEMORY_RESERVE_BPS", 500),
                descriptor_reserve_bps: env_basis_points("RUST_FD_RESERVE_BPS", 500),
                port_reserve_bps: env_basis_points("RUST_EPHEMERAL_PORT_RESERVE_BPS", 500),
                disk_reserve_bps: env_basis_points("RUST_REQUEST_SPOOL_DISK_RESERVE_BPS", 1000),
                inode_reserve_bps: env_basis_points("RUST_REQUEST_SPOOL_INODE_RESERVE_BPS", 500),
            }),
        }
    }

    #[cfg(test)]
    pub fn unconstrained_for_test() -> Self {
        Self {
            inner: Arc::new(Inner {
                cached: Mutex::new(CachedGlobalSample {
                    sampled_at: Instant::now()
                        .checked_sub(Duration::from_secs(1))
                        .unwrap_or_else(Instant::now),
                    rejection: None,
                }),
                sample_interval: Duration::from_millis(1),
                wait_timeout: Duration::from_secs(1),
                memory_reserve_bytes: 0,
                memory_reserve_bps: 0,
                descriptor_reserve_bps: 0,
                port_reserve_bps: 0,
                disk_reserve_bps: 0,
                inode_reserve_bps: 0,
            }),
        }
    }

    pub async fn wait_for_global_headroom(&self) -> Result<ResourceWait, CapacityFailure> {
        let started = Instant::now();
        loop {
            match self.global_rejection() {
                None => {
                    return Ok(ResourceWait {
                        waited: started.elapsed(),
                    })
                }
                Some(resource) if started.elapsed() >= self.inner.wait_timeout => {
                    return Err(CapacityFailure {
                        resource,
                        waited: started.elapsed(),
                    })
                }
                Some(_) => tokio::time::sleep(self.inner.sample_interval).await,
            }
        }
    }

    pub async fn wait_for_local_capacity(
        &self,
        path: &Path,
        additional_bytes: u64,
    ) -> Result<(ResourceWait, LocalCapacity), CapacityFailure> {
        let started = Instant::now();
        loop {
            if let Err(failure) = self.wait_for_global_headroom().await {
                return Err(CapacityFailure {
                    waited: started.elapsed(),
                    ..failure
                });
            }
            match self.local_capacity(path, additional_bytes) {
                Ok(capacity) => {
                    return Ok((
                        ResourceWait {
                            waited: started.elapsed(),
                        },
                        capacity,
                    ))
                }
                Err(resource) if started.elapsed() >= self.inner.wait_timeout => {
                    return Err(CapacityFailure {
                        resource,
                        waited: started.elapsed(),
                    })
                }
                Err(_) => tokio::time::sleep(self.inner.sample_interval).await,
            }
        }
    }

    pub fn local_capacity(
        &self,
        path: &Path,
        additional_bytes: u64,
    ) -> Result<LocalCapacity, ResourceKind> {
        let stats = statvfs(path).map_err(|_| ResourceKind::LocalDisk)?;
        let reserve_bytes = fraction(stats.total_bytes, self.inner.disk_reserve_bps);
        let writable_bytes = stats.free_bytes.saturating_sub(reserve_bytes);
        if writable_bytes < additional_bytes {
            return Err(ResourceKind::LocalDisk);
        }
        let reserve_inodes = fraction(stats.total_inodes, self.inner.inode_reserve_bps);
        let writable_inodes = stats.free_inodes.saturating_sub(reserve_inodes);
        if writable_inodes == 0 {
            return Err(ResourceKind::LocalInode);
        }
        Ok(LocalCapacity {
            free_bytes: stats.free_bytes,
            writable_bytes,
            free_inodes: stats.free_inodes,
            writable_inodes,
        })
    }

    fn global_rejection(&self) -> Option<ResourceKind> {
        let mut cached = match self.inner.cached.lock() {
            Ok(cached) => cached,
            Err(poisoned) => poisoned.into_inner(),
        };
        if cached.sampled_at.elapsed() < self.inner.sample_interval {
            return cached.rejection;
        }
        cached.sampled_at = Instant::now();
        cached.rejection = sample_global_rejection(&self.inner);
        cached.rejection
    }
}

fn sample_global_rejection(config: &Inner) -> Option<ResourceKind> {
    if let Some((current, limit)) = cgroup_memory() {
        let reserve = config
            .memory_reserve_bytes
            .max(fraction(limit, config.memory_reserve_bps));
        if current.saturating_add(reserve) >= limit {
            return Some(ResourceKind::Memory);
        }
    }
    if let Some((open, limit)) = file_descriptor_usage() {
        if open.saturating_add(fraction(limit, config.descriptor_reserve_bps)) >= limit {
            return Some(ResourceKind::FileDescriptor);
        }
        if let Some(connections) = tcp_connection_usage() {
            if connections.saturating_add(fraction(limit, config.descriptor_reserve_bps)) >= limit {
                return Some(ResourceKind::Connection);
            }
        }
    }
    if let Some((used, total)) = ephemeral_port_usage() {
        if used.saturating_add(fraction(total, config.port_reserve_bps)) >= total {
            return Some(ResourceKind::EphemeralPort);
        }
    }
    None
}

fn fraction(total: u64, basis_points: u64) -> u64 {
    total.saturating_mul(basis_points).saturating_add(9_999) / 10_000
}

fn cgroup_memory() -> Option<(u64, u64)> {
    for (current_path, limit_path) in [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ] {
        let Some(current) = read_u64(current_path) else {
            continue;
        };
        let Ok(limit_text) = fs::read_to_string(limit_path) else {
            continue;
        };
        let Ok(limit) = limit_text.trim().parse::<u64>() else {
            continue;
        };
        if limit > 0 && limit < (1_u64 << 60) {
            return Some((current, limit));
        }
    }
    None
}

fn file_descriptor_usage() -> Option<(u64, u64)> {
    let open = fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    let limits = fs::read_to_string("/proc/self/limits").ok()?;
    let limit = limits.lines().find_map(|line| {
        let rest = line.strip_prefix("Max open files")?.trim();
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some((open, limit))
}

fn tcp_connection_usage() -> Option<u64> {
    let mut count = 0_u64;
    let mut observed = false;
    for path in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(table) = fs::read_to_string(path) else {
            continue;
        };
        observed = true;
        count = count.saturating_add(
            table
                .lines()
                .skip(1)
                .filter(|line| line.split_whitespace().nth(3) != Some("0A"))
                .count() as u64,
        );
    }
    observed.then_some(count)
}

fn ephemeral_port_usage() -> Option<(u64, u64)> {
    let range = fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range").ok()?;
    let mut values = range
        .split_whitespace()
        .filter_map(|value| value.parse::<u16>().ok());
    let start = values.next()?;
    let end = values.next()?;
    let total = u64::from(end.saturating_sub(start)) + 1;
    let mut ports = HashSet::new();
    for path in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(table) = fs::read_to_string(path) else {
            continue;
        };
        for line in table.lines().skip(1) {
            let Some(local) = line.split_whitespace().nth(1) else {
                continue;
            };
            let Some(port_hex) = local.rsplit(':').next() else {
                continue;
            };
            let Ok(port) = u16::from_str_radix(port_hex, 16) else {
                continue;
            };
            if (start..=end).contains(&port) {
                ports.insert(port);
            }
        }
    }
    Some((ports.len() as u64, total))
}

#[derive(Clone, Copy)]
struct FileSystemStats {
    total_bytes: u64,
    free_bytes: u64,
    total_inodes: u64,
    free_inodes: u64,
}

#[cfg(unix)]
fn statvfs(path: &Path) -> io::Result<FileSystemStats> {
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path is a valid NUL-terminated string and stats points to writable memory.
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: statvfs initialized stats after returning success.
    let stats = unsafe { stats.assume_init() };
    let fragment_size = match stat_value(stats.f_frsize, 0) {
        0 => stat_value(stats.f_bsize, 0),
        value => value,
    };
    Ok(FileSystemStats {
        total_bytes: stat_value(stats.f_blocks, u64::MAX).saturating_mul(fragment_size),
        free_bytes: stat_value(stats.f_bavail, 0).saturating_mul(fragment_size),
        total_inodes: match stat_value(stats.f_files, 0) {
            0 => u64::MAX,
            value => value,
        },
        free_inodes: match stat_value(stats.f_files, 0) {
            0 => u64::MAX,
            _ => stat_value(stats.f_favail, 0),
        },
    })
}

fn stat_value<T>(value: T, fallback: u64) -> u64
where
    T: TryInto<u64>,
{
    value.try_into().unwrap_or(fallback)
}

#[cfg(not(unix))]
fn statvfs(_path: &Path) -> io::Result<FileSystemStats> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "filesystem capacity sampling is unavailable",
    ))
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}

fn env_basis_points(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value < 10_000)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_rounds_up_without_overflowing() {
        assert_eq!(fraction(100, 500), 5);
        assert_eq!(fraction(1, 500), 1);
        assert!(fraction(u64::MAX, 1_000) > 0);
    }

    #[test]
    fn local_capacity_is_resource_derived() {
        let governor = ResourceGovernor::unconstrained_for_test();
        let capacity = governor.local_capacity(Path::new("/tmp"), 0).unwrap();
        assert!(capacity.free_bytes > 0);
        assert!(capacity.writable_bytes <= capacity.free_bytes);
        assert!(capacity.writable_inodes <= capacity.free_inodes);
    }

    #[test]
    fn only_local_storage_exhaustion_uses_507() {
        assert_eq!(
            ResourceKind::LocalDisk.exhausted_status(),
            StatusCode::INSUFFICIENT_STORAGE
        );
        assert_eq!(
            ResourceKind::LocalInode.exhausted_status(),
            StatusCode::INSUFFICIENT_STORAGE
        );
        assert_eq!(
            ResourceKind::Memory.exhausted_status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
