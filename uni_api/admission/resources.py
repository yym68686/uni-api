from __future__ import annotations

import math
import os
import resource
from dataclasses import dataclass
from pathlib import Path


_KIB = 1024
_DEFAULT_BASE_ACTIVE = 64
_DEFAULT_REFERENCE_CPU_MILLICORES = 465
_DEFAULT_REFERENCE_CPU_WEIGHT = 55
_DEFAULT_MIN_TOTAL = 1000
_DEFAULT_BURST_MULTIPLIER = 2.0
_DEFAULT_CONNECTION_MEMORY_BYTES = 128 * _KIB
_DEFAULT_ACTIVE_EXTRA_MEMORY_BYTES = 384 * _KIB
_DEFAULT_CONTROL_MEMORY_RATIO = 0.125
_DEFAULT_FD_RESERVE_MIN = 256
_DEFAULT_FD_RESERVE_RATIO = 0.05
_DEFAULT_OUTBOUND_FDS_PER_ACTIVE = 2
_DEFAULT_TRANSPORT_HEADROOM_MIN = 8
_DEFAULT_TRANSPORT_HEADROOM_RATIO = 0.10
_DEFAULT_EPHEMERAL_PORT_UTILIZATION = 0.80
_DEFAULT_PER_REQUEST_MEMORY_MIN = 8 * 1024 * 1024
_DEFAULT_PER_REQUEST_PROCESS_RATIO = 0.125
_DEFAULT_PER_REQUEST_FAIR_SHARE_MULTIPLIER = 32
_DEFAULT_LARGE_REQUEST_PROCESS_RATIO = 0.25
_DEFAULT_JSON_REQUEST_HEADROOM_MULTIPLIER = 1


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_int_environment(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int_environment(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _positive_float_environment(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bool_environment(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _ratio_environment(name: str, default: float) -> float:
    value = _positive_float_environment(name, default)
    if value >= 1:
        raise ValueError(f"{name} must be below one")
    return value


def current_cgroup_v2_root(
    root: str | Path = "/sys/fs/cgroup",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> Path:
    root_path = Path(root)
    content = _read_text(Path(proc_cgroup))
    if not content:
        return root_path
    for line in content.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        if hierarchy != "0" or controllers:
            continue
        relative_path = relative.strip().lstrip("/")
        if not relative_path:
            return root_path
        candidate = root_path / relative_path
        if candidate.exists():
            return candidate
    return root_path


def current_cgroup_v1_root(
    controller: str,
    root: str | Path = "/sys/fs/cgroup",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> Path:
    """Resolve this process' directory for one cgroup v1 controller."""

    normalized = str(controller or "").strip()
    if not normalized:
        raise ValueError("cgroup v1 controller name is required")

    root_path = Path(root)
    fallback = root_path / normalized
    content = _read_text(Path(proc_cgroup))
    if not content:
        return fallback

    for line in content.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _hierarchy, controllers, relative = fields
        names = [name.strip() for name in controllers.split(",") if name.strip()]
        if normalized not in names:
            continue

        relative_path = relative.strip().lstrip("/")
        for mount_name in (controllers.strip(), normalized):
            if not mount_name:
                continue
            candidate = root_path / mount_name
            if relative_path:
                candidate = candidate / relative_path
            if candidate.exists():
                return candidate

        candidate = fallback
        if relative_path:
            candidate = candidate / relative_path
        return candidate

    return fallback


def cgroup_cpu_quota_millicores(
    root: str | Path = "/sys/fs/cgroup",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> int | None:
    """Return the startup cgroup CPU quota, or None when it is unlimited."""

    root_path = current_cgroup_v2_root(root, proc_cgroup)
    cpu_max = _read_text(root_path / "cpu.max")
    if cpu_max:
        fields = cpu_max.split()
        if len(fields) >= 2 and fields[0] != "max":
            try:
                quota = int(fields[0])
                period = int(fields[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return max(1, quota * 1000 // period)

    v1_root = current_cgroup_v1_root("cpu", root, proc_cgroup)
    quota_text = _read_text(v1_root / "cpu.cfs_quota_us")
    period_text = _read_text(v1_root / "cpu.cfs_period_us")
    if quota_text is None or period_text is None:
        return None
    try:
        quota = int(quota_text)
        period = int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota * 1000 // period)


def cgroup_cpu_weight(
    root: str | Path = "/sys/fs/cgroup",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> int | None:
    value = _read_text(current_cgroup_v2_root(root, proc_cgroup) / "cpu.weight")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def cgroup_cpu_shares_millicores(
    root: str | Path = "/sys/fs/cgroup",
    proc_cgroup: str | Path = "/proc/self/cgroup",
) -> int | None:
    """Translate a cgroup v1 CPU-share request into Kubernetes millicores."""

    value = _read_text(
        current_cgroup_v1_root("cpu", root, proc_cgroup) / "cpu.shares"
    )
    if value is None:
        return None
    try:
        shares = int(value)
    except ValueError:
        return None
    if shares <= 0:
        return None
    return max(1, shares * 1000 // 1024)


def process_cpu_affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count()
        return count if count is not None and count > 0 else None


def startup_cpu_worker_count(
    *,
    executor_groups: int = 3,
    minimum_per_group: int = 4,
) -> int:
    """Size each CPU-offload executor once for the new Pod.

    Three independent executors currently serve request decoding, JSON work,
    and upstream response decoding. Dividing the detected entitlement between
    them avoids creating three full CPU-count pools while allowing a larger
    replacement Pod to use more than the historical four threads per pool.
    """

    if executor_groups <= 0 or minimum_per_group <= 0:
        raise ValueError("CPU worker sizing inputs must be positive")
    configured_cpu = _optional_positive_int_environment(
        "REQUEST_ADMISSION_CPU_MILLICORES"
    )
    quota = cgroup_cpu_quota_millicores()
    weight = cgroup_cpu_weight()
    shares = cgroup_cpu_shares_millicores()
    affinity = process_cpu_affinity_count()
    if configured_cpu is not None:
        entitlement = configured_cpu
    elif quota is not None:
        entitlement = quota
    elif weight is not None:
        entitlement = (
            _DEFAULT_REFERENCE_CPU_MILLICORES
            * weight
            // _DEFAULT_REFERENCE_CPU_WEIGHT
        )
    elif shares is not None:
        entitlement = shares
    elif affinity is not None:
        entitlement = affinity * 1000
    else:
        entitlement = _DEFAULT_REFERENCE_CPU_MILLICORES
    if affinity is not None:
        entitlement = min(entitlement, affinity * 1000)
    cpu_count = max(1, math.ceil(entitlement / 1000))
    return max(minimum_per_group, math.ceil(cpu_count / executor_groups))


def startup_per_request_memory_limit(
    *,
    process_memory_capacity_bytes: int,
    active_limit: int,
    minimum_bytes: int = _DEFAULT_PER_REQUEST_MEMORY_MIN,
    maximum_process_ratio: float = _DEFAULT_PER_REQUEST_PROCESS_RATIO,
    fair_share_multiplier: int = _DEFAULT_PER_REQUEST_FAIR_SHARE_MULTIPLIER,
) -> int:
    """Derive one request's retained-memory ceiling from the new Pod.

    A request may burst above its equal share, but cannot own more than one
    eighth of process memory. The fair-share multiplier prevents proportional
    CPU/connection growth from granting a single client an ever-larger slice.
    The shared parent governor remains the aggregate authority.
    """

    if process_memory_capacity_bytes <= 0 or active_limit <= 0:
        raise ValueError("per-request memory sizing inputs must be positive")
    if minimum_bytes <= 0 or fair_share_multiplier <= 0:
        raise ValueError("per-request memory sizing inputs must be positive")
    if not 0 < maximum_process_ratio < 1:
        raise ValueError("maximum_process_ratio must be between zero and one")
    process_ceiling = max(
        1,
        math.floor(process_memory_capacity_bytes * maximum_process_ratio),
    )
    fair_burst = max(
        1,
        process_memory_capacity_bytes
        * fair_share_multiplier
        // active_limit,
    )
    return min(process_ceiling, max(minimum_bytes, fair_burst))


def startup_large_request_memory_limit(
    *,
    process_memory_capacity_bytes: int,
    normal_request_limit_bytes: int,
    product_wire_limit_bytes: int,
    raw_memory_multiplier: int,
    maximum_process_ratio: float = _DEFAULT_LARGE_REQUEST_PROCESS_RATIO,
    headroom_multiplier: int = _DEFAULT_JSON_REQUEST_HEADROOM_MULTIPLIER,
) -> int:
    """Derive a bounded large-JSON allowance from the effective cgroup.

    The product wire limit remains stable on sufficiently sized runtimes. A
    smaller runtime degrades to its safe per-request ceiling instead of
    allowing one request to consume an unbounded share of process memory.
    """

    values = (
        process_memory_capacity_bytes,
        normal_request_limit_bytes,
        product_wire_limit_bytes,
        raw_memory_multiplier,
        headroom_multiplier,
    )
    if any(value <= 0 for value in values):
        raise ValueError("large request memory sizing inputs must be positive")
    if not 0 < maximum_process_ratio < 1:
        raise ValueError("maximum_process_ratio must be between zero and one")
    process_ceiling = max(
        1,
        math.floor(process_memory_capacity_bytes * maximum_process_ratio),
    )
    product_target = product_wire_limit_bytes * (
        raw_memory_multiplier + headroom_multiplier
    )
    return max(
        normal_request_limit_bytes,
        min(product_target, process_ceiling),
    )


def process_nofile_soft_limit() -> int | None:
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    if soft == resource.RLIM_INFINITY or soft <= 0:
        return None
    return int(soft)


def process_open_fd_count() -> int | None:
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            return sum(1 for _entry in directory.iterdir())
        except OSError:
            continue
    return None


def ephemeral_port_count(
    range_path: str | Path = "/proc/sys/net/ipv4/ip_local_port_range",
    reserved_path: str | Path = "/proc/sys/net/ipv4/ip_local_reserved_ports",
) -> int | None:
    raw_range = _read_text(Path(range_path))
    if not raw_range:
        return None
    fields = raw_range.split()
    if len(fields) != 2:
        return None
    try:
        lower, upper = (int(field) for field in fields)
    except ValueError:
        return None
    if lower <= 0 or upper < lower:
        return None

    reserved: set[int] = set()
    raw_reserved = _read_text(Path(reserved_path)) or ""
    for item in raw_reserved.split(","):
        item = item.strip()
        if not item:
            continue
        start_text, separator, end_text = item.partition("-")
        try:
            start = int(start_text)
            end = int(end_text) if separator else start
        except ValueError:
            continue
        start = max(lower, start)
        end = min(upper, end)
        if start <= end:
            reserved.update(range(start, end + 1))
    return max(0, upper - lower + 1 - len(reserved))


def tcp_socket_port_occupancy(
    path: str | Path = "/proc/net/sockstat",
) -> int | None:
    """Return current TCP sockets plus TIME_WAIT entries in this netns."""

    content = _read_text(Path(path))
    if not content:
        return None
    for line in content.splitlines():
        name, separator, values = line.partition(":")
        if not separator or name.strip() != "TCP":
            continue
        fields = values.split()
        counters: dict[str, int] = {}
        for index in range(0, len(fields) - 1, 2):
            try:
                counters[fields[index]] = int(fields[index + 1])
            except ValueError:
                continue
        return max(0, counters.get("inuse", 0)) + max(
            0,
            counters.get("tw", 0),
        )
    return None


def kernel_somaxconn(
    path: str | Path = "/proc/sys/net/core/somaxconn",
) -> int | None:
    value = _read_text(Path(path))
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True, slots=True)
class StartupConcurrencyEnvelope:
    active_limit: int
    waiter_limit: int
    total_limit: int
    uvicorn_limit_concurrency: int
    uvicorn_backlog: int
    active_sizing_source: str
    cpu_millicores: int | None
    cpu_weight: int | None
    cpu_affinity_count: int | None
    cpu_sizing_source: str
    cpu_active_limit: int
    resource_active_limit: int
    memory_available_bytes: int | None
    memory_control_budget_bytes: int | None
    nofile_soft_limit: int | None
    open_fds: int | None
    ephemeral_port_count: int | None
    ephemeral_port_occupancy: int | None
    ephemeral_active_limit: int | None


def startup_active_limit(
    *,
    cpu_millicores: int | None,
    cpu_weight: int | None = None,
    cpu_affinity_count: int | None = None,
    base_active: int = _DEFAULT_BASE_ACTIVE,
    reference_cpu_millicores: int = _DEFAULT_REFERENCE_CPU_MILLICORES,
    reference_cpu_weight: int = _DEFAULT_REFERENCE_CPU_WEIGHT,
    explicit_max: int | None = None,
) -> int:
    """Scale active request slots from this new Pod's startup CPU envelope."""

    if base_active <= 0 or reference_cpu_millicores <= 0:
        raise ValueError("startup active-limit inputs must be positive")
    if reference_cpu_weight <= 0:
        raise ValueError("reference CPU weight must be positive")
    if explicit_max is not None and explicit_max <= 0:
        raise ValueError("explicit active limit cap must be positive")

    entitlement: int | None = None
    if cpu_millicores is not None and cpu_millicores > 0:
        # A finite quota is an absolute CPU ceiling and is more meaningful
        # than the relative scheduling weight exposed alongside it.
        entitlement = cpu_millicores
    elif cpu_weight is not None and cpu_weight > 0:
        # Fugue encodes a request (not a hard CPU limit) as cgroup weight. The
        # reference pair is calibrated from the known 465m/weight=55 Pod.
        entitlement = max(
            1,
            reference_cpu_millicores * cpu_weight // reference_cpu_weight,
        )
    elif cpu_affinity_count is not None and cpu_affinity_count > 0:
        # Outside a constrained cgroup the schedulable CPU set is the only
        # concrete startup allocation, so do not collapse a large bare-metal
        # server back to the 64-slot fallback.
        entitlement = cpu_affinity_count * 1000

    if entitlement is not None:
        if cpu_affinity_count is not None and cpu_affinity_count > 0:
            entitlement = min(entitlement, cpu_affinity_count * 1000)
        active = max(1, base_active * entitlement // reference_cpu_millicores)
    else:
        active = base_active
    if explicit_max is not None:
        active = min(active, explicit_max)
    return active


def _transport_headroom(
    total: int,
    *,
    minimum: int,
    ratio: float,
) -> int:
    return max(minimum, math.ceil(total * ratio))


def calculate_startup_concurrency(
    *,
    cpu_millicores: int | None,
    cpu_weight: int | None,
    cpu_affinity_count: int | None,
    memory_available_bytes: int | None,
    nofile_soft_limit: int | None,
    open_fds: int | None,
    ephemeral_ports: int | None,
    ephemeral_ports_in_use: int | None = None,
    somaxconn: int | None,
    cpu_sizing_source: str | None = None,
    requested_active: int | None = None,
    requested_waiters: int | None = None,
    requested_total: int | None = None,
    base_active: int = _DEFAULT_BASE_ACTIVE,
    reference_cpu_millicores: int = _DEFAULT_REFERENCE_CPU_MILLICORES,
    reference_cpu_weight: int = _DEFAULT_REFERENCE_CPU_WEIGHT,
    explicit_max_active: int | None = None,
    cpu_bound_active: bool = True,
    minimum_total: int = _DEFAULT_MIN_TOTAL,
    burst_multiplier: float = _DEFAULT_BURST_MULTIPLIER,
    connection_memory_bytes: int = _DEFAULT_CONNECTION_MEMORY_BYTES,
    active_extra_memory_bytes: int = _DEFAULT_ACTIVE_EXTRA_MEMORY_BYTES,
    control_memory_ratio: float = _DEFAULT_CONTROL_MEMORY_RATIO,
    fd_reserve_min: int = _DEFAULT_FD_RESERVE_MIN,
    fd_reserve_ratio: float = _DEFAULT_FD_RESERVE_RATIO,
    outbound_fds_per_active: int = _DEFAULT_OUTBOUND_FDS_PER_ACTIVE,
    transport_headroom_min: int = _DEFAULT_TRANSPORT_HEADROOM_MIN,
    transport_headroom_ratio: float = _DEFAULT_TRANSPORT_HEADROOM_RATIO,
    ephemeral_port_utilization: float = _DEFAULT_EPHEMERAL_PORT_UTILIZATION,
) -> StartupConcurrencyEnvelope:
    """Build one immutable process envelope from the new Pod's resources.

    Active work is prioritized over queued work. Automatic waiters are trimmed
    before active slots whenever file-descriptor or control-memory budgets bind.
    Explicit unsafe active/waiter settings fail fast instead of silently
    pretending that the process can honor them.
    """

    positive_values = {
        "base_active": base_active,
        "reference_cpu_millicores": reference_cpu_millicores,
        "reference_cpu_weight": reference_cpu_weight,
        "minimum_total": minimum_total,
        "connection_memory_bytes": connection_memory_bytes,
        "active_extra_memory_bytes": active_extra_memory_bytes,
        "fd_reserve_min": fd_reserve_min,
        "outbound_fds_per_active": outbound_fds_per_active,
        "transport_headroom_min": transport_headroom_min,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise ValueError("startup concurrency sizing inputs must be positive")
    for name, value in {
        "control_memory_ratio": control_memory_ratio,
        "fd_reserve_ratio": fd_reserve_ratio,
        "transport_headroom_ratio": transport_headroom_ratio,
        "ephemeral_port_utilization": ephemeral_port_utilization,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between zero and one")
    if burst_multiplier < 1:
        raise ValueError("burst_multiplier must be at least one")

    cpu_limit = startup_active_limit(
        cpu_millicores=cpu_millicores,
        cpu_weight=cpu_weight,
        cpu_affinity_count=cpu_affinity_count,
        base_active=base_active,
        reference_cpu_millicores=reference_cpu_millicores,
        reference_cpu_weight=reference_cpu_weight,
        explicit_max=explicit_max_active,
    )
    cpu_sizing_source = cpu_sizing_source or (
        "quota"
        if cpu_millicores is not None and cpu_millicores > 0
        else "weight"
        if cpu_weight is not None and cpu_weight > 0
        else "affinity"
        if cpu_affinity_count is not None and cpu_affinity_count > 0
        else "fallback"
    )
    port_active_limit: int | None = None
    if ephemeral_ports is not None:
        if ephemeral_ports <= 0:
            raise ValueError("no ephemeral ports are available for upstream traffic")
        unoccupied_ephemeral_ports = max(
            0,
            ephemeral_ports - max(0, ephemeral_ports_in_use or 0),
        )
        if unoccupied_ephemeral_ports <= 0:
            raise ValueError("no unoccupied ephemeral ports are available")
        port_active_limit = max(
            1,
            math.floor(
                unoccupied_ephemeral_ports
                * ephemeral_port_utilization
                / outbound_fds_per_active
            ),
        )
    memory_control_budget: int | None = None
    if memory_available_bytes is not None:
        if memory_available_bytes <= 0:
            raise ValueError("no startup memory headroom is available")
        memory_control_budget = math.floor(
            memory_available_bytes * control_memory_ratio
        )
    fd_reserve = 0
    if nofile_soft_limit is not None and nofile_soft_limit > 0:
        adaptive_absolute_reserve = min(
            fd_reserve_min,
            max(1, nofile_soft_limit // 2),
        )
        fd_reserve = max(
            adaptive_absolute_reserve,
            math.ceil(nofile_soft_limit * fd_reserve_ratio),
        )
    baseline_open_fds = max(0, open_fds or 0)

    def fits(active: int, waiters: int) -> bool:
        total = active + waiters
        headroom = _transport_headroom(
            total,
            minimum=transport_headroom_min,
            ratio=transport_headroom_ratio,
        )
        server_limit = total + headroom
        if nofile_soft_limit is not None:
            required_fds = (
                baseline_open_fds
                + fd_reserve
                + outbound_fds_per_active * active
                + server_limit
            )
            if required_fds > nofile_soft_limit:
                return False
        if memory_control_budget is not None:
            required_memory = (
                connection_memory_bytes * server_limit
                + active_extra_memory_bytes * active
            )
            if required_memory > memory_control_budget:
                return False
        return True

    def largest_fitting(upper: int, predicate) -> int:
        low, high = 0, max(0, upper)
        while low < high:
            middle = (low + high + 1) // 2
            if predicate(middle):
                low = middle
            else:
                high = middle - 1
        return low

    # Find the whole-request ceiling from concrete connection resources. The
    # loose arithmetic bounds only bound the binary search; ``fits`` applies
    # the exact transport headroom, FD, and control-memory accounting.
    resource_upper_bounds: list[int] = []
    if port_active_limit is not None:
        resource_upper_bounds.append(port_active_limit)
    if memory_control_budget is not None:
        resource_upper_bounds.append(
            memory_control_budget
            // (connection_memory_bytes + active_extra_memory_bytes)
        )
    if nofile_soft_limit is not None:
        available_fds = max(
            0,
            nofile_soft_limit - baseline_open_fds - fd_reserve,
        )
        resource_upper_bounds.append(
            available_fds // (outbound_fds_per_active + 1)
        )
    if resource_upper_bounds:
        resource_search_limit = min(resource_upper_bounds)
    else:
        # Unconstrained developer hosts still need a finite envelope. This
        # fallback is never used when any memory, FD, or port bound is known.
        resource_search_limit = max(
            minimum_total,
            math.ceil(cpu_limit * burst_multiplier),
        )
    resource_active_limit = largest_fitting(
        resource_search_limit,
        lambda value: fits(value, 0),
    )
    if resource_active_limit <= 0:
        raise ValueError("Pod startup resources cannot safely host one active request")

    if requested_active is not None:
        target_active = requested_active
    elif cpu_bound_active:
        target_active = cpu_limit
    else:
        # Whole requests own memory and transport state. CPU entitlement is
        # enforced separately, only while bounded CPU callbacks execute.
        target_active = resource_active_limit
    if (
        requested_active is not None
        and explicit_max_active is not None
        and requested_active > explicit_max_active
    ):
        raise ValueError("requested active limit exceeds explicit maximum")
    if explicit_max_active is not None:
        target_active = min(target_active, explicit_max_active)
    if target_active <= 0:
        raise ValueError("requested active limit must be positive")
    if requested_total is not None:
        if requested_active is not None and requested_active > requested_total:
            raise ValueError("REQUEST_ADMISSION_TOTAL_LIMIT is below active limit")
        target_active = min(target_active, requested_total)

    active = min(target_active, resource_active_limit)
    if requested_active is not None and active != requested_active:
        raise ValueError("REQUEST_ADMISSION_ACTIVE_LIMIT exceeds startup resources")

    if requested_total is not None:
        total_waiter_ceiling = requested_total - active
    elif requested_waiters is not None:
        total_waiter_ceiling = requested_waiters
    else:
        total_waiter_ceiling = max(
            0,
            max(minimum_total, math.ceil(active * burst_multiplier)) - active,
        )
    desired_waiters = (
        requested_waiters if requested_waiters is not None else total_waiter_ceiling
    )
    if desired_waiters < 0:
        raise ValueError("requested waiter limit cannot be negative")
    if desired_waiters > total_waiter_ceiling:
        raise ValueError("request admission envelope exceeds requested total limit")

    waiters = largest_fitting(desired_waiters, lambda value: fits(active, value))
    if requested_waiters is not None and waiters != requested_waiters:
        raise ValueError("REQUEST_ADMISSION_WAITER_LIMIT exceeds startup resources")

    total = active + waiters
    headroom = _transport_headroom(
        total,
        minimum=transport_headroom_min,
        ratio=transport_headroom_ratio,
    )
    uvicorn_limit = total + headroom
    backlog_target = max(1, waiters + headroom)
    backlog = min(backlog_target, somaxconn) if somaxconn else backlog_target
    return StartupConcurrencyEnvelope(
        active_limit=active,
        waiter_limit=waiters,
        total_limit=total,
        uvicorn_limit_concurrency=uvicorn_limit,
        uvicorn_backlog=max(1, backlog),
        active_sizing_source=(
            "explicit"
            if requested_active is not None
            else "cpu"
            if cpu_bound_active
            else "resource"
        ),
        cpu_millicores=cpu_millicores,
        cpu_weight=cpu_weight,
        cpu_affinity_count=cpu_affinity_count,
        cpu_sizing_source=cpu_sizing_source,
        cpu_active_limit=cpu_limit,
        resource_active_limit=resource_active_limit,
        memory_available_bytes=memory_available_bytes,
        memory_control_budget_bytes=memory_control_budget,
        nofile_soft_limit=nofile_soft_limit,
        open_fds=open_fds,
        ephemeral_port_count=ephemeral_ports,
        ephemeral_port_occupancy=ephemeral_ports_in_use,
        ephemeral_active_limit=port_active_limit,
    )


def startup_concurrency_from_environment(
    *,
    memory_available_bytes: int | None,
    honor_request_count_overrides: bool = True,
) -> StartupConcurrencyEnvelope:
    configured_cpu = _optional_positive_int_environment(
        "REQUEST_ADMISSION_CPU_MILLICORES"
    )
    requested_active = (
        _optional_positive_int_environment("REQUEST_ADMISSION_ACTIVE_LIMIT")
        if honor_request_count_overrides
        else None
    )
    requested_waiters = (
        _nonnegative_int_environment("REQUEST_ADMISSION_WAITER_LIMIT")
        if honor_request_count_overrides
        else None
    )
    requested_total = (
        _optional_positive_int_environment("REQUEST_ADMISSION_TOTAL_LIMIT")
        if honor_request_count_overrides
        else None
    )
    explicit_max = (
        _optional_positive_int_environment(
            "REQUEST_ADMISSION_MAX_ACTIVE_LIMIT"
        )
        if honor_request_count_overrides
        else None
    )
    cpu_bound_active = (
        _bool_environment("REQUEST_ADMISSION_CPU_BOUND_ACTIVE", False)
        if honor_request_count_overrides
        else False
    )
    base_active = _positive_int_environment(
        "REQUEST_ADMISSION_BASE_ACTIVE_LIMIT",
        _DEFAULT_BASE_ACTIVE,
    )
    reference_cpu = _positive_int_environment(
        "REQUEST_ADMISSION_REFERENCE_CPU_MILLICORES",
        _DEFAULT_REFERENCE_CPU_MILLICORES,
    )
    reference_weight = _positive_int_environment(
        "REQUEST_ADMISSION_REFERENCE_CPU_WEIGHT",
        _DEFAULT_REFERENCE_CPU_WEIGHT,
    )
    minimum_total = _positive_int_environment(
        "REQUEST_ADMISSION_MIN_TOTAL_LIMIT",
        _DEFAULT_MIN_TOTAL,
    )
    burst_multiplier = _positive_float_environment(
        "REQUEST_ADMISSION_BURST_MULTIPLIER",
        _DEFAULT_BURST_MULTIPLIER,
    )
    connection_memory = _positive_int_environment(
        "REQUEST_ADMISSION_CONNECTION_MEMORY_BYTES",
        _DEFAULT_CONNECTION_MEMORY_BYTES,
    )
    active_extra_memory = _positive_int_environment(
        "REQUEST_ADMISSION_ACTIVE_EXTRA_MEMORY_BYTES",
        _DEFAULT_ACTIVE_EXTRA_MEMORY_BYTES,
    )
    control_memory_ratio = _ratio_environment(
        "REQUEST_ADMISSION_CONTROL_MEMORY_RATIO",
        _DEFAULT_CONTROL_MEMORY_RATIO,
    )
    fd_reserve_min = _positive_int_environment(
        "REQUEST_ADMISSION_FD_RESERVE_MIN",
        _DEFAULT_FD_RESERVE_MIN,
    )
    fd_reserve_ratio = _ratio_environment(
        "REQUEST_ADMISSION_FD_RESERVE_RATIO",
        _DEFAULT_FD_RESERVE_RATIO,
    )
    outbound_fds = _positive_int_environment(
        "REQUEST_ADMISSION_OUTBOUND_FDS_PER_ACTIVE",
        _DEFAULT_OUTBOUND_FDS_PER_ACTIVE,
    )
    headroom_min = _positive_int_environment(
        "REQUEST_ADMISSION_TRANSPORT_HEADROOM_MIN",
        _DEFAULT_TRANSPORT_HEADROOM_MIN,
    )
    headroom_ratio = _ratio_environment(
        "REQUEST_ADMISSION_TRANSPORT_HEADROOM_RATIO",
        _DEFAULT_TRANSPORT_HEADROOM_RATIO,
    )
    port_utilization = _ratio_environment(
        "REQUEST_ADMISSION_EPHEMERAL_PORT_UTILIZATION",
        _DEFAULT_EPHEMERAL_PORT_UTILIZATION,
    )

    detected_quota = cgroup_cpu_quota_millicores()
    detected_weight = cgroup_cpu_weight()
    detected_shares = cgroup_cpu_shares_millicores()
    if configured_cpu is not None:
        cpu_millicores = configured_cpu
        cpu_weight = None
        cpu_source = "configured"
    elif detected_quota is not None:
        cpu_millicores = detected_quota
        cpu_weight = None
        cpu_source = "quota"
    elif detected_weight is not None:
        cpu_millicores = None
        cpu_weight = detected_weight
        cpu_source = "weight"
    elif detected_shares is not None:
        cpu_millicores = detected_shares
        cpu_weight = None
        cpu_source = "shares"
    else:
        cpu_millicores = None
        cpu_weight = None
        cpu_source = "affinity"

    return calculate_startup_concurrency(
        cpu_millicores=cpu_millicores,
        cpu_weight=cpu_weight,
        cpu_affinity_count=process_cpu_affinity_count(),
        memory_available_bytes=memory_available_bytes,
        nofile_soft_limit=process_nofile_soft_limit(),
        open_fds=process_open_fd_count(),
        ephemeral_ports=ephemeral_port_count(),
        ephemeral_ports_in_use=tcp_socket_port_occupancy(),
        somaxconn=kernel_somaxconn(),
        cpu_sizing_source=cpu_source,
        requested_active=requested_active,
        requested_waiters=requested_waiters,
        requested_total=requested_total,
        base_active=base_active,
        reference_cpu_millicores=reference_cpu,
        reference_cpu_weight=reference_weight,
        explicit_max_active=explicit_max,
        cpu_bound_active=cpu_bound_active,
        minimum_total=minimum_total,
        burst_multiplier=burst_multiplier,
        connection_memory_bytes=connection_memory,
        active_extra_memory_bytes=active_extra_memory,
        control_memory_ratio=control_memory_ratio,
        fd_reserve_min=fd_reserve_min,
        fd_reserve_ratio=fd_reserve_ratio,
        outbound_fds_per_active=outbound_fds,
        transport_headroom_min=headroom_min,
        transport_headroom_ratio=headroom_ratio,
        ephemeral_port_utilization=port_utilization,
    )
