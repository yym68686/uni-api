use std::collections::HashSet;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde_json::{json, Value};

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
    memory_soft_limit_bytes: Option<u64>,
    memory_guard_bytes: u64,
    memory_guard_bps: u64,
    memory_fallback_budget_bytes: u64,
    descriptor_reserve_bps: u64,
    port_reserve_bps: u64,
    disk_reserve_bps: u64,
    inode_reserve_bps: u64,
    reserved_memory_bytes: AtomicU64,
    shared_memory_ledger: Option<Arc<SharedMemoryReservationLedger>>,
}

struct SharedMemoryReservationLedger {
    file: Mutex<File>,
}

const SHARED_MEMORY_FIELDS: usize = 6;
const SHARED_MEMORY_LEDGER_BYTES: u64 = (SHARED_MEMORY_FIELDS * 8) as u64;

#[allow(dead_code)] // Numeric slots are a cross-language ledger ABI.
#[derive(Clone, Copy)]
enum SharedMemoryCategory {
    ParsedBody = 1,
    SerializedBody = 2,
    TransportBuffer = 3,
    ResponseBuffer = 4,
    Other = 5,
}

impl SharedMemoryReservationLedger {
    fn open(path: &Path, reset: bool) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let ledger = Self {
            file: Mutex::new(file),
        };
        ledger.with_locked_file(|file| {
            if reset || file.metadata()?.len() < SHARED_MEMORY_LEDGER_BYTES {
                Self::write_locked(file, [0; SHARED_MEMORY_FIELDS])?;
            }
            Ok(())
        })?;
        Ok(ledger)
    }

    fn with_locked_file<T>(
        &self,
        operation: impl FnOnce(&mut File) -> io::Result<T>,
    ) -> io::Result<T> {
        let mut file = match self.file.lock() {
            Ok(file) => file,
            Err(poisoned) => poisoned.into_inner(),
        };
        #[cfg(unix)]
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let result = operation(&mut file);
        #[cfg(unix)]
        let unlock_result = if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) } != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        };
        match (result, unlock_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    fn read_locked(file: &mut File) -> io::Result<[u64; SHARED_MEMORY_FIELDS]> {
        file.seek(SeekFrom::Start(0))?;
        let mut payload = [0_u8; SHARED_MEMORY_FIELDS * 8];
        file.read_exact(&mut payload)?;
        let mut values = [0_u64; SHARED_MEMORY_FIELDS];
        for (index, value) in values.iter_mut().enumerate() {
            let offset = index * 8;
            *value = u64::from_le_bytes(
                payload[offset..offset + 8]
                    .try_into()
                    .expect("shared memory field width is fixed"),
            );
        }
        Ok(values)
    }

    fn write_locked(file: &mut File, values: [u64; SHARED_MEMORY_FIELDS]) -> io::Result<()> {
        file.seek(SeekFrom::Start(0))?;
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
        file.set_len(SHARED_MEMORY_LEDGER_BYTES)?;
        file.flush()
    }

    fn total(&self) -> io::Result<u64> {
        self.with_locked_file(|file| Ok(Self::read_locked(file)?[0]))
    }

    fn try_reserve(
        &self,
        category: SharedMemoryCategory,
        bytes: u64,
        maximum_total: u64,
    ) -> io::Result<bool> {
        self.with_locked_file(|file| {
            let mut values = Self::read_locked(file)?;
            let before = values[0];
            let Some(after) = before.checked_add(bytes) else {
                return Ok(false);
            };
            if after > maximum_total {
                return Ok(false);
            }
            values[0] = after;
            let category_index = category as usize;
            values[category_index] = values[category_index].saturating_add(bytes);
            Self::write_locked(file, values)?;
            Ok(true)
        })
    }

    fn release(&self, category: SharedMemoryCategory, bytes: u64) -> io::Result<()> {
        self.with_locked_file(|file| {
            let mut values = Self::read_locked(file)?;
            let category_index = category as usize;
            values[0] = values[0].checked_sub(bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "shared memory reservation ledger underflow",
                )
            })?;
            values[category_index] =
                values[category_index].checked_sub(bytes).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "shared memory reservation category underflow",
                    )
                })?;
            Self::write_locked(file, values)
        })
    }
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
        let shared_memory_path = PathBuf::from(
            std::env::var("UNI_API_SHARED_MEMORY_RESERVATION_PATH")
                .unwrap_or_else(|_| "/tmp/uni-api-shared-memory-reservation-v1".to_string()),
        );
        let shared_memory_ledger = SharedMemoryReservationLedger::open(&shared_memory_path, true)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to initialize shared memory reservation ledger {}: {error}",
                    shared_memory_path.display()
                )
            });
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
                memory_soft_limit_bytes: env_optional_u64("MEMORY_SOFT_LIMIT_BYTES"),
                memory_guard_bytes: env_u64_allow_zero("MEMORY_GUARD_BYTES", 512 * 1024 * 1024),
                memory_guard_bps: env_ratio_basis_points("MEMORY_GUARD_RATIO", 0.25),
                memory_fallback_budget_bytes: env_u64(
                    "MEMORY_FALLBACK_BUDGET_BYTES",
                    256 * 1024 * 1024,
                ),
                descriptor_reserve_bps: env_basis_points("RUST_FD_RESERVE_BPS", 500),
                port_reserve_bps: env_basis_points("RUST_EPHEMERAL_PORT_RESERVE_BPS", 500),
                disk_reserve_bps: env_basis_points("RUST_REQUEST_SPOOL_DISK_RESERVE_BPS", 1000),
                inode_reserve_bps: env_basis_points("RUST_REQUEST_SPOOL_INODE_RESERVE_BPS", 500),
                reserved_memory_bytes: AtomicU64::new(0),
                shared_memory_ledger: Some(Arc::new(shared_memory_ledger)),
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
                memory_soft_limit_bytes: None,
                memory_guard_bytes: 0,
                memory_guard_bps: 0,
                memory_fallback_budget_bytes: u64::MAX,
                descriptor_reserve_bps: 0,
                port_reserve_bps: 0,
                disk_reserve_bps: 0,
                inode_reserve_bps: 0,
                reserved_memory_bytes: AtomicU64::new(0),
                shared_memory_ledger: None,
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

    pub async fn reserve_memory_capacity(
        &self,
        additional_bytes: u64,
    ) -> Result<(ResourceWait, MemoryReservation), CapacityFailure> {
        self.reserve_memory_capacity_for(SharedMemoryCategory::ParsedBody, additional_bytes)
            .await
    }

    async fn reserve_memory_capacity_for(
        &self,
        category: SharedMemoryCategory,
        additional_bytes: u64,
    ) -> Result<(ResourceWait, MemoryReservation), CapacityFailure> {
        let started = Instant::now();
        loop {
            let reserved = self.inner.reserved_memory_bytes.load(Ordering::Acquire);
            let maximum_total = cgroup_memory()
                .map(|(current, limit)| {
                    memory_soft_limit(&self.inner, limit).saturating_sub(current)
                })
                .unwrap_or(self.inner.memory_fallback_budget_bytes);
            let shared_reserved = self
                .inner
                .shared_memory_ledger
                .as_ref()
                .and_then(|ledger| ledger.total().ok())
                .unwrap_or(reserved);
            let available = shared_reserved.saturating_add(additional_bytes) <= maximum_total;
            let local_claimed = available
                && self
                    .inner
                    .reserved_memory_bytes
                    .compare_exchange(
                        reserved,
                        reserved.saturating_add(additional_bytes),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok();
            let shared_claimed = local_claimed
                && self
                    .inner
                    .shared_memory_ledger
                    .as_ref()
                    .is_none_or(|ledger| {
                        ledger
                            .try_reserve(category, additional_bytes, maximum_total)
                            .unwrap_or(false)
                    });
            if shared_claimed {
                return Ok((
                    ResourceWait {
                        waited: started.elapsed(),
                    },
                    MemoryReservation {
                        inner: self.inner.clone(),
                        bytes: additional_bytes,
                        category,
                    },
                ));
            }
            if local_claimed {
                self.inner
                    .reserved_memory_bytes
                    .fetch_sub(additional_bytes, Ordering::AcqRel);
            }
            if started.elapsed() >= self.inner.wait_timeout {
                return Err(CapacityFailure {
                    resource: ResourceKind::Memory,
                    waited: started.elapsed(),
                });
            }
            tokio::time::sleep(self.inner.sample_interval).await;
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

    pub fn observability_snapshot(&self) -> Value {
        let reserved = self
            .inner
            .shared_memory_ledger
            .as_ref()
            .and_then(|ledger| ledger.total().ok())
            .unwrap_or_else(|| self.inner.reserved_memory_bytes.load(Ordering::Acquire));
        let (memory_current, memory_limit) = cgroup_memory().unwrap_or((0, 0));
        let (open_fds, fd_limit) = file_descriptor_usage().unwrap_or((0, 0));
        let (ephemeral_ports_used, ephemeral_ports_total) =
            ephemeral_port_usage().unwrap_or((0, 0));
        json!({
            "reserved_memory_bytes":reserved,
            "cgroup_memory_current_bytes":memory_current,
            "cgroup_memory_limit_bytes":memory_limit,
            "open_file_descriptors":open_fds,
            "file_descriptor_limit":fd_limit,
            "tcp_connections":tcp_connection_usage().unwrap_or(0),
            "ephemeral_ports_used":ephemeral_ports_used,
            "ephemeral_ports_total":ephemeral_ports_total,
            "rejection_resource":self.global_rejection().map(ResourceKind::as_str),
        })
    }
}

pub struct MemoryReservation {
    inner: Arc<Inner>,
    bytes: u64,
    category: SharedMemoryCategory,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if let Some(ledger) = &self.inner.shared_memory_ledger {
            if let Err(error) = ledger.release(self.category, self.bytes) {
                eprintln!("shared memory reservation ledger release failed: {error}");
            }
        }
        self.inner
            .reserved_memory_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

fn sample_global_rejection(config: &Inner) -> Option<ResourceKind> {
    if let Some((current, limit)) = cgroup_memory() {
        let reserved = config
            .shared_memory_ledger
            .as_ref()
            .and_then(|ledger| ledger.total().ok())
            .unwrap_or_else(|| config.reserved_memory_bytes.load(Ordering::Acquire));
        if current.saturating_add(reserved) >= memory_soft_limit(config, limit) {
            return Some(ResourceKind::Memory);
        }
    } else {
        let reserved = config
            .shared_memory_ledger
            .as_ref()
            .and_then(|ledger| ledger.total().ok())
            .unwrap_or_else(|| config.reserved_memory_bytes.load(Ordering::Acquire));
        if reserved >= config.memory_fallback_budget_bytes {
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

fn memory_soft_limit(config: &Inner, limit: u64) -> u64 {
    if let Some(configured) = config.memory_soft_limit_bytes {
        return configured.min(limit).max(1);
    }
    let absolute_guard = config.memory_guard_bytes.min(limit / 2);
    let guard = absolute_guard.max(fraction(limit, config.memory_guard_bps));
    limit.saturating_sub(guard.min(limit.saturating_sub(1)))
}

fn cgroup_memory() -> Option<(u64, u64)> {
    for (current_path, limit_path, high_path) in [
        (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.max",
            Some("/sys/fs/cgroup/memory.high"),
        ),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            None,
        ),
    ] {
        let Some(current) = read_u64(current_path) else {
            continue;
        };
        let Ok(limit_text) = fs::read_to_string(limit_path) else {
            continue;
        };
        let Some(mut limit) = parse_memory_limit(limit_text.trim(), system_memory_total_bytes())
        else {
            continue;
        };
        if let Some(high) = high_path.and_then(read_u64) {
            if high > 0 && high < (1_u64 << 60) {
                limit = limit.min(high);
            }
        }
        if limit > 0 && limit < (1_u64 << 60) {
            return Some((current, limit));
        }
    }
    None
}

fn parse_memory_limit(value: &str, unbounded_fallback: Option<u64>) -> Option<u64> {
    let parsed = value.trim().parse::<u64>().ok();
    match parsed {
        Some(limit) if limit > 0 && limit < (1_u64 << 60) => Some(limit),
        _ if value.trim() == "max" || parsed.is_some() => unbounded_fallback,
        _ => None,
    }
}

fn system_memory_total_bytes() -> Option<u64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| {
            let kib = line.strip_prefix("MemTotal:")?.split_whitespace().next()?;
            kib.parse::<u64>().ok()?.checked_mul(1024)
        })
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

fn env_u64_allow_zero(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_optional_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn env_ratio_basis_points(name: &str, default: f64) -> u64 {
    let ratio = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(0.0, 0.95);
    (ratio * 10_000.0).round() as u64
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
    fn unbounded_cgroup_memory_uses_host_capacity() {
        let fallback = Some(8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("max", fallback), fallback);
        assert_eq!(
            parse_memory_limit(&(1_u64 << 60).to_string(), fallback),
            fallback
        );
        assert_eq!(
            parse_memory_limit("2147483648", fallback),
            Some(2_147_483_648)
        );
        assert_eq!(parse_memory_limit("invalid", fallback), None);
    }

    #[test]
    fn rust_soft_limit_matches_python_guard_policy() {
        let production = Inner {
            cached: Mutex::new(CachedGlobalSample {
                sampled_at: Instant::now(),
                rejection: None,
            }),
            sample_interval: Duration::from_millis(1),
            wait_timeout: Duration::from_secs(1),
            memory_soft_limit_bytes: None,
            memory_guard_bytes: 512 * 1024 * 1024,
            memory_guard_bps: 2_500,
            memory_fallback_budget_bytes: 256 * 1024 * 1024,
            descriptor_reserve_bps: 0,
            port_reserve_bps: 0,
            disk_reserve_bps: 0,
            inode_reserve_bps: 0,
            reserved_memory_bytes: AtomicU64::new(0),
            shared_memory_ledger: None,
        };
        assert_eq!(
            memory_soft_limit(&production, 256 * 1024 * 1024),
            128 * 1024 * 1024
        );
        assert_eq!(
            memory_soft_limit(&production, 512 * 1024 * 1024),
            256 * 1024 * 1024
        );
        assert_eq!(
            memory_soft_limit(&production, 4032 * 1024 * 1024),
            3024 * 1024 * 1024
        );
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

    #[tokio::test]
    async fn memory_reservations_are_released_without_request_slots() {
        let governor = ResourceGovernor::unconstrained_for_test();
        let (_, reservation) = governor.reserve_memory_capacity(4096).await.unwrap();
        assert_eq!(
            governor.inner.reserved_memory_bytes.load(Ordering::Acquire),
            4096
        );
        drop(reservation);
        assert_eq!(
            governor.inner.reserved_memory_bytes.load(Ordering::Acquire),
            0
        );
    }

    #[test]
    fn independent_shared_ledgers_observe_the_same_total() {
        let path =
            std::env::temp_dir().join(format!("uni-api-memory-ledger-test-{}", std::process::id()));
        let first = SharedMemoryReservationLedger::open(&path, true).unwrap();
        let second = SharedMemoryReservationLedger::open(&path, false).unwrap();
        assert!(first
            .try_reserve(SharedMemoryCategory::ParsedBody, 400, 1_000)
            .unwrap());
        assert_eq!(second.total().unwrap(), 400);
        assert!(!second
            .try_reserve(SharedMemoryCategory::TransportBuffer, 601, 1_000)
            .unwrap());
        assert!(second
            .try_reserve(SharedMemoryCategory::TransportBuffer, 600, 1_000)
            .unwrap());
        assert_eq!(first.total().unwrap(), 1_000);
        second
            .release(SharedMemoryCategory::TransportBuffer, 600)
            .unwrap();
        first
            .release(SharedMemoryCategory::ParsedBody, 400)
            .unwrap();
        assert_eq!(first.total().unwrap(), 0);
        let _ = fs::remove_file(path);
    }
}
