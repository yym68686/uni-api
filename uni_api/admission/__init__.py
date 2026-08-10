"""Bounded, cancellation-safe admission primitives."""

from uni_api.admission.core import (
    AdmissionLease,
    AdmissionRejected,
    BoundedAdmissionGate,
    LargeBodyCapacityExhausted,
    PendingBodyReservation,
    RequestAdmissionController,
    RequestAdmissionLease,
    RequestBodyBudgetExhausted,
    RequestBodyTooLarge,
    TemporaryResponseBytesReservation,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    get_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.cpu import (
    CPUPhaseLimiter,
    cpu_phase_snapshot,
    run_cpu_phase,
)
from uni_api.admission.network import (
    AdaptiveNetworkGovernor,
    AdaptiveNetworkSnapshot,
    NetworkResourceLease,
)
from uni_api.admission.observability import (
    Admission503ResponseWriteOutcome,
    LargeBodyAdmissionDecision,
    LargeBodyHolderSnapshot,
    ResponseBufferEvent,
    RequestBodyObservation,
)

__all__ = [
    "AdmissionLease",
    "Admission503ResponseWriteOutcome",
    "AdmissionRejected",
    "AdaptiveNetworkGovernor",
    "AdaptiveNetworkSnapshot",
    "BoundedAdmissionGate",
    "CPUPhaseLimiter",
    "LargeBodyCapacityExhausted",
    "LargeBodyAdmissionDecision",
    "LargeBodyHolderSnapshot",
    "NetworkResourceLease",
    "ResponseBufferEvent",
    "PendingBodyReservation",
    "RequestAdmissionController",
    "RequestAdmissionLease",
    "RequestBodyBudgetExhausted",
    "RequestBodyObservation",
    "RequestBodyTooLarge",
    "TemporaryResponseBytesReservation",
    "UpstreamResponseBudgetExhausted",
    "bind_request_admission_lease",
    "get_request_admission_lease",
    "cpu_phase_snapshot",
    "reset_request_admission_lease",
    "run_cpu_phase",
]
