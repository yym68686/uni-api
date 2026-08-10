from pathlib import Path

import pytest

import uni_api.admission.resources as resources_module
from uni_api.admission.resources import (
    calculate_startup_concurrency,
    cgroup_cpu_quota_millicores,
    cgroup_cpu_shares_millicores,
    cgroup_cpu_weight,
    current_cgroup_v1_root,
    current_cgroup_v2_root,
    ephemeral_port_count,
    startup_active_limit,
    startup_concurrency_from_environment,
    startup_cpu_worker_count,
    startup_large_request_memory_limit,
    startup_per_request_memory_limit,
    tcp_socket_port_occupancy,
)


def test_reads_cgroup_v2_cpu_quota(tmp_path: Path):
    (tmp_path / "cpu.max").write_text("46500 100000\n", encoding="ascii")
    assert cgroup_cpu_quota_millicores(tmp_path) == 465


def test_unlimited_cgroup_cpu_uses_fallback(tmp_path: Path):
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="ascii")
    assert cgroup_cpu_quota_millicores(tmp_path) is None


def test_reads_cgroup_v2_cpu_weight(tmp_path: Path):
    (tmp_path / "cpu.weight").write_text("55\n", encoding="ascii")
    assert cgroup_cpu_weight(tmp_path) == 55


def test_resolves_non_namespaced_process_cgroup(tmp_path: Path):
    nested = tmp_path / "kubepods.slice" / "pod.scope"
    nested.mkdir(parents=True)
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text("0::/kubepods.slice/pod.scope\n", encoding="ascii")
    (nested / "cpu.max").write_text("50000 100000\n", encoding="ascii")

    assert current_cgroup_v2_root(tmp_path, proc_cgroup) == nested
    assert cgroup_cpu_quota_millicores(tmp_path, proc_cgroup) == 500


def test_reads_cgroup_v1_cpu_quota(tmp_path: Path):
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("100000\n", encoding="ascii")
    (cpu / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")
    assert cgroup_cpu_quota_millicores(tmp_path) == 1000


def test_resolves_nested_combined_cgroup_v1_cpu_controller(tmp_path: Path):
    nested = tmp_path / "cpu,cpuacct" / "kubepods" / "pod" / "container"
    nested.mkdir(parents=True)
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text(
        "2:cpu,cpuacct:/kubepods/pod/container\n"
        "3:memory:/kubepods/pod/container\n",
        encoding="ascii",
    )
    (nested / "cpu.cfs_quota_us").write_text("50000\n", encoding="ascii")
    (nested / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")

    assert current_cgroup_v1_root("cpu", tmp_path, proc_cgroup) == nested
    assert cgroup_cpu_quota_millicores(tmp_path, proc_cgroup) == 500


def test_reads_cgroup_v1_cpu_shares_as_millicores(tmp_path: Path):
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.shares").write_text("2048\n", encoding="ascii")
    assert cgroup_cpu_shares_millicores(tmp_path) == 2000


@pytest.mark.parametrize(
    ("cpu", "expected"),
    [(None, 64), (232, 31), (465, 64), (1000, 137), (2000, 275), (128000, 17617)],
)
def test_startup_active_scales_without_an_arbitrary_default_cap(cpu, expected):
    assert startup_active_limit(cpu_millicores=cpu) == expected


def test_cpu_weight_and_affinity_are_real_startup_bounds():
    assert startup_active_limit(cpu_millicores=None, cpu_weight=110) == 128
    assert (
        startup_active_limit(
            cpu_millicores=128000,
            cpu_weight=None,
            cpu_affinity_count=8,
        )
        == 1101
    )
    assert (
        startup_active_limit(
            cpu_millicores=None,
            cpu_weight=None,
            cpu_affinity_count=128,
        )
        == 17617
    )


def test_finite_cpu_quota_is_not_artificially_reduced_by_default_weight():
    assert (
        startup_active_limit(
            cpu_millicores=128000,
            cpu_weight=100,
            cpu_affinity_count=128,
        )
        == 17617
    )


def test_cpu_executor_workers_scale_once_across_three_groups(monkeypatch):
    monkeypatch.setenv("REQUEST_ADMISSION_CPU_MILLICORES", "128000")
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 128)
    assert startup_cpu_worker_count() == 43


def test_per_request_memory_limit_uses_process_fraction_and_fair_share():
    gib = 1024**3
    assert (
        startup_per_request_memory_limit(
            process_memory_capacity_bytes=3 * gib,
            active_limit=64,
        )
        == 384 * 1024**2
    )
    assert (
        startup_per_request_memory_limit(
            process_memory_capacity_bytes=96 * gib,
            active_limit=11292,
        )
        == 292_112_547
    )


def test_large_request_memory_limit_honors_product_target_on_sized_runtime():
    mib = 1024**2
    assert (
        startup_large_request_memory_limit(
            process_memory_capacity_bytes=3432 * mib,
            normal_request_limit_bytes=429 * mib,
            product_wire_limit_bytes=128 * mib,
            raw_memory_multiplier=5,
        )
        == 768 * mib
    )


def test_live_envelope_admits_the_rejected_request_with_one_large_slot():
    mib = 1024**2
    body_limit = startup_large_request_memory_limit(
        process_memory_capacity_bytes=3432 * mib,
        normal_request_limit_bytes=429 * mib,
        product_wire_limit_bytes=128 * mib,
        raw_memory_multiplier=5,
    )
    weighted_large_threshold = 128 * mib * 2
    incident_estimate = 269_605_381

    assert body_limit == 768 * mib
    assert body_limit // 6 == 128 * mib
    assert weighted_large_threshold < incident_estimate < body_limit


def test_large_request_memory_limit_contracts_to_cgroup_fraction():
    mib = 1024**2
    assert (
        startup_large_request_memory_limit(
            process_memory_capacity_bytes=2 * 1024 * mib,
            normal_request_limit_bytes=256 * mib,
            product_wire_limit_bytes=128 * mib,
            raw_memory_multiplier=5,
        )
        == 512 * mib
    )


def _envelope(**overrides):
    values = {
        "cpu_millicores": None,
        "cpu_weight": 55,
        "cpu_affinity_count": 8,
        "memory_available_bytes": 3 * 1024**3,
        "nofile_soft_limit": 1_048_576,
        "open_fds": 20,
        "ephemeral_ports": 28_232,
        "somaxconn": 4096,
    }
    values.update(overrides)
    return calculate_startup_concurrency(**values)


def test_current_fugue_shape_preserves_1000_request_burst_envelope():
    envelope = _envelope()
    assert envelope.cpu_sizing_source == "weight"
    assert envelope.active_limit == 64
    assert envelope.waiter_limit == 936
    assert envelope.total_limit == 1000
    assert envelope.uvicorn_limit_concurrency == 1100
    assert envelope.uvicorn_backlog == 1036


def test_resource_separated_mode_sizes_whole_requests_without_cpu_cap():
    envelope = _envelope(cpu_bound_active=False)

    assert envelope.active_sizing_source == "resource"
    assert envelope.cpu_active_limit == 64
    assert envelope.resource_active_limit == 749
    assert envelope.active_limit == 749
    assert envelope.waiter_limit == 1
    assert envelope.total_limit == 750


def test_resource_separated_mode_does_not_inherit_cpu_search_ceiling():
    envelope = _envelope(
        cpu_weight=1,
        memory_available_bytes=96 * 1024**3,
        cpu_bound_active=False,
    )

    assert envelope.cpu_active_limit == 1
    assert envelope.ephemeral_active_limit == 11292
    assert envelope.resource_active_limit == 11292
    assert envelope.active_limit == 11292


def test_128_cpu_128_gib_shape_is_not_capped_at_100():
    envelope = _envelope(
        cpu_millicores=128000,
        cpu_weight=None,
        cpu_affinity_count=128,
        memory_available_bytes=96 * 1024**3,
    )
    assert envelope.cpu_active_limit == 17617
    assert envelope.ephemeral_active_limit == 11292
    assert envelope.active_limit == 11292
    assert envelope.waiter_limit == 11292
    assert envelope.total_limit == 22584
    assert envelope.uvicorn_limit_concurrency == 24843


def test_default_waiters_trim_before_active_when_fd_budget_binds():
    envelope = _envelope(
        nofile_soft_limit=1000,
        open_fds=10,
        ephemeral_ports=None,
        memory_available_bytes=None,
        fd_reserve_min=100,
        fd_reserve_ratio=0.01,
    )
    assert envelope.active_limit == 64
    assert 0 < envelope.waiter_limit < 936
    required = 10 + 100 + 2 * envelope.active_limit
    required += envelope.uvicorn_limit_concurrency
    assert required <= 1000


def test_low_but_usable_nofile_limit_scales_reserve_instead_of_failing():
    envelope = _envelope(
        nofile_soft_limit=256,
        open_fds=4,
        ephemeral_ports=None,
        memory_available_bytes=None,
        minimum_total=1,
    )
    assert 1 <= envelope.active_limit < 64
    assert envelope.uvicorn_limit_concurrency <= 256


def test_control_memory_budget_can_reduce_active_safely():
    envelope = _envelope(
        memory_available_bytes=64 * 1024**2,
        nofile_soft_limit=None,
        ephemeral_ports=None,
        minimum_total=1,
    )
    assert 1 <= envelope.active_limit < 64
    assert envelope.memory_control_budget_bytes == 8 * 1024**2


def test_zero_memory_headroom_or_ephemeral_ports_fails_closed():
    with pytest.raises(ValueError, match="memory headroom"):
        _envelope(memory_available_bytes=0)
    with pytest.raises(ValueError, match="ephemeral ports"):
        _envelope(ephemeral_ports=0)


def test_explicit_unsafe_active_and_waiter_limits_fail_fast():
    with pytest.raises(ValueError, match="ACTIVE_LIMIT exceeds startup resources"):
        _envelope(
            requested_active=1000,
            nofile_soft_limit=1000,
            open_fds=10,
            ephemeral_ports=None,
            memory_available_bytes=None,
            fd_reserve_min=100,
            fd_reserve_ratio=0.01,
        )
    with pytest.raises(ValueError, match="WAITER_LIMIT exceeds startup resources"):
        _envelope(
            requested_waiters=10000,
            nofile_soft_limit=2000,
            open_fds=10,
            ephemeral_ports=None,
            memory_available_bytes=None,
            fd_reserve_min=100,
            fd_reserve_ratio=0.01,
        )


def test_explicit_active_limit_wins_over_cpu_but_not_safety(monkeypatch):
    monkeypatch.setenv("REQUEST_ADMISSION_ACTIVE_LIMIT", "72")
    monkeypatch.setattr(resources_module, "cgroup_cpu_quota_millicores", lambda: None)
    monkeypatch.setattr(resources_module, "cgroup_cpu_weight", lambda: 55)
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 8)
    monkeypatch.setattr(resources_module, "process_nofile_soft_limit", lambda: 1_000_000)
    monkeypatch.setattr(resources_module, "process_open_fd_count", lambda: 10)
    monkeypatch.setattr(resources_module, "ephemeral_port_count", lambda: 28000)
    monkeypatch.setattr(resources_module, "kernel_somaxconn", lambda: 4096)
    envelope = startup_concurrency_from_environment(
        memory_available_bytes=3 * 1024**3
    )
    assert envelope.active_limit == 72


def test_explicit_total_caps_automatic_active_without_becoming_a_default_cap():
    envelope = _envelope(
        cpu_millicores=128000,
        cpu_weight=None,
        cpu_affinity_count=128,
        memory_available_bytes=96 * 1024**3,
        requested_total=1000,
    )
    assert envelope.active_limit == 1000
    assert envelope.waiter_limit == 0
    assert envelope.total_limit == 1000


def test_environment_cpu_weight_scales_beyond_100(monkeypatch):
    monkeypatch.delenv("REQUEST_ADMISSION_ACTIVE_LIMIT", raising=False)
    monkeypatch.delenv("REQUEST_ADMISSION_MAX_ACTIVE_LIMIT", raising=False)
    monkeypatch.setattr(resources_module, "cgroup_cpu_quota_millicores", lambda: None)
    monkeypatch.setattr(resources_module, "cgroup_cpu_weight", lambda: 110)
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 8)
    monkeypatch.setattr(resources_module, "process_nofile_soft_limit", lambda: 1_000_000)
    monkeypatch.setattr(resources_module, "process_open_fd_count", lambda: 10)
    monkeypatch.setattr(resources_module, "ephemeral_port_count", lambda: 28000)
    monkeypatch.setattr(resources_module, "kernel_somaxconn", lambda: 4096)
    envelope = startup_concurrency_from_environment(
        memory_available_bytes=3 * 1024**3
    )
    assert envelope.active_sizing_source == "resource"
    assert envelope.cpu_active_limit == 128
    assert envelope.active_limit > envelope.cpu_active_limit


def test_environment_can_restore_legacy_cpu_bound_request_admission(monkeypatch):
    monkeypatch.setenv("REQUEST_ADMISSION_CPU_BOUND_ACTIVE", "true")
    monkeypatch.setattr(resources_module, "cgroup_cpu_quota_millicores", lambda: None)
    monkeypatch.setattr(resources_module, "cgroup_cpu_weight", lambda: 55)
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 8)
    monkeypatch.setattr(resources_module, "process_nofile_soft_limit", lambda: 1_000_000)
    monkeypatch.setattr(resources_module, "process_open_fd_count", lambda: 10)
    monkeypatch.setattr(resources_module, "ephemeral_port_count", lambda: 28000)
    monkeypatch.setattr(resources_module, "tcp_socket_port_occupancy", lambda: 0)
    monkeypatch.setattr(resources_module, "kernel_somaxconn", lambda: 4096)

    envelope = startup_concurrency_from_environment(
        memory_available_bytes=3 * 1024**3
    )

    assert envelope.active_sizing_source == "cpu"
    assert envelope.active_limit == 64
    assert envelope.waiter_limit == 936


def test_configured_cpu_entitlement_supports_uncalibrated_standalone_hosts(
    monkeypatch,
):
    monkeypatch.setenv("REQUEST_ADMISSION_CPU_MILLICORES", "128000")
    monkeypatch.setattr(resources_module, "cgroup_cpu_quota_millicores", lambda: None)
    monkeypatch.setattr(resources_module, "cgroup_cpu_weight", lambda: 100)
    monkeypatch.setattr(resources_module, "cgroup_cpu_shares_millicores", lambda: None)
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 128)
    monkeypatch.setattr(resources_module, "process_nofile_soft_limit", lambda: 1_000_000)
    monkeypatch.setattr(resources_module, "process_open_fd_count", lambda: 10)
    monkeypatch.setattr(resources_module, "ephemeral_port_count", lambda: 28000)
    monkeypatch.setattr(resources_module, "tcp_socket_port_occupancy", lambda: 0)
    monkeypatch.setattr(resources_module, "kernel_somaxconn", lambda: 4096)
    envelope = startup_concurrency_from_environment(
        memory_available_bytes=96 * 1024**3
    )
    assert envelope.cpu_sizing_source == "configured"
    assert envelope.cpu_active_limit == 17617


def test_invalid_explicit_active_limit_fails_fast(monkeypatch):
    monkeypatch.setenv("REQUEST_ADMISSION_ACTIVE_LIMIT", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        startup_concurrency_from_environment(memory_available_bytes=None)


def test_weighted_startup_ignores_legacy_request_count_overrides(monkeypatch):
    monkeypatch.setenv("REQUEST_ADMISSION_ACTIVE_LIMIT", "invalid")
    monkeypatch.setenv("REQUEST_ADMISSION_WAITER_LIMIT", "invalid")
    monkeypatch.setenv("REQUEST_ADMISSION_TOTAL_LIMIT", "invalid")
    monkeypatch.setenv("REQUEST_ADMISSION_MAX_ACTIVE_LIMIT", "invalid")
    monkeypatch.setenv("REQUEST_ADMISSION_CPU_BOUND_ACTIVE", "invalid")
    monkeypatch.setattr(
        resources_module,
        "cgroup_cpu_quota_millicores",
        lambda: 1000,
    )
    monkeypatch.setattr(resources_module, "cgroup_cpu_weight", lambda: None)
    monkeypatch.setattr(resources_module, "process_cpu_affinity_count", lambda: 1)
    monkeypatch.setattr(resources_module, "process_nofile_soft_limit", lambda: 10000)
    monkeypatch.setattr(resources_module, "process_open_fd_count", lambda: 10)
    monkeypatch.setattr(resources_module, "ephemeral_port_count", lambda: 28000)
    monkeypatch.setattr(resources_module, "tcp_socket_port_occupancy", lambda: 0)
    monkeypatch.setattr(resources_module, "kernel_somaxconn", lambda: 4096)

    envelope = startup_concurrency_from_environment(
        memory_available_bytes=3 * 1024**3,
        honor_request_count_overrides=False,
    )

    assert envelope.active_sizing_source == "resource"


def test_ephemeral_port_count_excludes_reserved_ranges(tmp_path: Path):
    port_range = tmp_path / "range"
    reserved = tmp_path / "reserved"
    port_range.write_text("10000 10009\n", encoding="ascii")
    reserved.write_text("10001-10003,10008\n", encoding="ascii")
    assert ephemeral_port_count(port_range, reserved) == 6


def test_tcp_socket_occupancy_includes_inuse_and_time_wait(tmp_path: Path):
    sockstat = tmp_path / "sockstat"
    sockstat.write_text(
        "sockets: used 10\nTCP: inuse 7 orphan 0 tw 13 alloc 8 mem 1\n",
        encoding="ascii",
    )
    assert tcp_socket_port_occupancy(sockstat) == 20


def test_ephemeral_active_limit_subtracts_current_netns_occupancy():
    envelope = _envelope(
        ephemeral_ports=1000,
        ephemeral_ports_in_use=200,
        outbound_fds_per_active=2,
    )
    assert envelope.ephemeral_active_limit == 320
