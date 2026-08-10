import asyncio

from uni_api.admission.network import AdaptiveNetworkGovernor


def _governor(*, ports=100, occupied=98):
    return AdaptiveNetworkGovernor(
        nofile_supplier=lambda: 10_000,
        open_fds_supplier=lambda: 10,
        ephemeral_ports_supplier=lambda: ports,
        ephemeral_occupancy_supplier=lambda: occupied,
        fd_reserve_min=0,
        fd_reserve_ratio=0,
        ephemeral_port_utilization=0.99,
        sample_cache_seconds=0,
        wait_poll_seconds=0.01,
    )


def test_network_governor_waits_on_live_port_headroom_without_count_capacity():
    async def scenario():
        governor = _governor()
        first = governor.try_acquire()
        assert first is not None

        second_task = asyncio.create_task(governor.acquire())
        await asyncio.sleep(0.02)
        snapshot = governor.snapshot()
        assert snapshot.pending_connection_attempts == 1
        assert snapshot.waiting_connection_attempts == 1

        await first.release()
        second = await asyncio.wait_for(second_task, timeout=1)
        assert second.wait_ms > 0
        await second.release()

        snapshot = governor.snapshot()
        assert snapshot.pending_connection_attempts == 0
        assert snapshot.waiting_connection_attempts == 0
        assert snapshot.acquired_total == 2

    asyncio.run(scenario())


def test_network_governor_cancellation_does_not_leak_a_reservation():
    async def scenario():
        governor = _governor()
        holder = governor.try_acquire()
        assert holder is not None
        waiter = asyncio.create_task(governor.acquire())
        await asyncio.sleep(0.02)
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        await holder.release()

        snapshot = governor.snapshot()
        assert snapshot.pending_connection_attempts == 0
        assert snapshot.waiting_connection_attempts == 0
        assert snapshot.cancelled_total == 1

    asyncio.run(scenario())


def test_release_keeps_a_cached_charge_without_forcing_procfs_resampling():
    now = 0.0
    samples = 0

    def open_fds():
        nonlocal samples
        samples += 1
        return 10

    governor = AdaptiveNetworkGovernor(
        nofile_supplier=lambda: 100,
        open_fds_supplier=open_fds,
        ephemeral_ports_supplier=lambda: 100,
        ephemeral_occupancy_supplier=lambda: 0,
        fd_reserve_min=10,
        fd_reserve_ratio=0,
        ephemeral_port_utilization=0.8,
        sample_cache_seconds=60,
        clock=lambda: now,
    )

    lease = governor.try_acquire()
    assert lease is not None
    asyncio.run(lease.release())
    cached = governor.snapshot()
    assert samples == 1
    assert cached.completed_connection_attempts_since_sample == 1
    assert cached.fd_headroom == 79
    assert cached.ephemeral_port_headroom == 79

    now = 61
    refreshed = governor.snapshot()
    assert samples == 2
    assert refreshed.completed_connection_attempts_since_sample == 0
    assert refreshed.fd_headroom == 80
    assert refreshed.ephemeral_port_headroom == 80


def test_inbound_guard_uses_fd_headroom_instead_of_connection_count():
    open_fds = 90
    governor = AdaptiveNetworkGovernor(
        nofile_supplier=lambda: 100,
        open_fds_supplier=lambda: open_fds,
        ephemeral_ports_supplier=lambda: None,
        ephemeral_occupancy_supplier=lambda: None,
        fd_reserve_min=10,
        fd_reserve_ratio=0,
        sample_cache_seconds=0,
    )

    assert governor.allow_inbound_connection() is False
    open_fds = 89
    assert governor.allow_inbound_connection() is True


def test_inbound_guard_accounts_for_accept_bursts_inside_sample_window():
    governor = AdaptiveNetworkGovernor(
        nofile_supplier=lambda: 100,
        open_fds_supplier=lambda: 80,
        ephemeral_ports_supplier=lambda: None,
        ephemeral_occupancy_supplier=lambda: None,
        fd_reserve_min=10,
        fd_reserve_ratio=0,
        sample_cache_seconds=60,
    )

    assert all(governor.allow_inbound_connection() for _ in range(10))
    assert governor.allow_inbound_connection() is False
    snapshot = governor.snapshot()
    assert snapshot.inbound_accepts_since_sample == 10
    assert snapshot.fd_headroom == 0
