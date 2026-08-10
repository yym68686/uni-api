from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum

try:
    from uni_api import _uni_api_native
except ImportError:
    _uni_api_native = None


DEFAULT_JSON_RAW_MEMORY_MULTIPLIER = 5
DEFAULT_JSON_TOKEN_MEMORY_BYTES = 1024
DEFAULT_JSON_MAX_DEPTH = 128
DEFAULT_JSON_MAX_SCALAR_BYTES = 4096
DEFAULT_JSON_MAX_ESTIMATED_BYTES = 256 * 1024 * 1024
_TEXT_SCAN_CHUNK_CHARACTERS = 256 * 1024
_JSON_STRING_SPECIAL = re.compile(br'["\\]')
_JSON_OUTSIDE_SPECIAL = re.compile(br'["{}\[\]\x20\x09\x0a\x0d,:]')
_NATIVE_INTEGER_MAX = (1 << 64) - 1
_REQUESTED_JSON_MEMORY_BACKEND = os.getenv(
    "UNI_API_JSON_GUARD_BACKEND",
    "auto",
).strip().lower()
if _REQUESTED_JSON_MEMORY_BACKEND not in {"auto", "python", "rust"}:
    raise ValueError(
        "UNI_API_JSON_GUARD_BACKEND must be auto, python, or rust"
    )
if _REQUESTED_JSON_MEMORY_BACKEND == "rust" and _uni_api_native is None:
    raise RuntimeError(
        "UNI_API_JSON_GUARD_BACKEND=rust requires uni_api._uni_api_native"
    )
_USE_NATIVE_JSON_MEMORY = _uni_api_native is not None and (
    _REQUESTED_JSON_MEMORY_BACKEND != "python"
)


class JSONMemoryComplexityReason(StrEnum):
    MAX_DEPTH = "max_depth"
    MAX_SCALAR_BYTES = "max_scalar_bytes"
    MAX_ESTIMATED_BYTES = "max_estimated_bytes"


class JSONMemoryComplexityTriggerPhase(StrEnum):
    CHUNK_RAW_CHARGE = "chunk_raw_charge"
    STRUCTURAL_ITEM_SCAN = "structural_item_scan"
    DEPTH_SCAN = "depth_scan"
    SCALAR_SCAN = "scalar_scan"


@dataclass(frozen=True, slots=True)
class JSONMemoryComplexityObservation:
    """Body-free primitives captured at the exact rejection decision.

    The scanner preserves its existing whole-chunk raw-memory charge. The
    cumulative ``raw_bytes`` and low-cardinality trigger phase identify the
    rejected input frame without adding per-byte work to accepted requests.
    """

    schema_version: int
    reason: JSONMemoryComplexityReason
    trigger_phase: JSONMemoryComplexityTriggerPhase
    raw_bytes: int
    structural_item_count: int
    depth: int
    peak_depth: int
    scalar_bytes: int
    estimated_bytes: int
    configured_limit: int
    max_depth: int
    max_scalar_bytes: int
    max_estimated_bytes: int
    raw_memory_multiplier: int
    structural_item_memory_bytes: int


class JSONMemoryComplexityError(ValueError):
    """A JSON document exceeds a finite structural/memory envelope."""

    def __init__(
        self,
        message: str,
        *,
        observation: JSONMemoryComplexityObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.observation = observation


@dataclass(frozen=True, slots=True)
class JSONMemorySnapshot:
    raw_bytes: int
    tokens: int
    depth: int
    peak_depth: int
    scalar_bytes: int
    estimated_bytes: int
    raw_memory_multiplier: int
    structural_item_memory_bytes: int


def json_memory_backend() -> str:
    return "rust" if _USE_NATIVE_JSON_MEMORY else "python"


def json_memory_native_available() -> bool:
    return _uni_api_native is not None


class IncrementalJSONMemoryEstimator:
    """Estimate JSON materialization memory with O(1) retained state.

    Raw-byte multipliers alone are unsafe for object-dense JSON: a three-byte
    ``{}`` can become a dict, list slot, and (for typed routes) validation
    objects.  This scanner therefore charges both source bytes and every
    string/scalar/container token.  The defaults deliberately overestimate
    measured CPython 3.11 + Pydantic peaks while still allowing large single
    strings such as base64 image inputs.

    The scanner is not a JSON validator.  Malformed documents keep their
    existing FastAPI error behavior; we only reject unreasonably deep/large
    structures before a parser can materialize them.
    """

    def __init__(
        self,
        *,
        raw_memory_multiplier: int = DEFAULT_JSON_RAW_MEMORY_MULTIPLIER,
        token_memory_bytes: int = DEFAULT_JSON_TOKEN_MEMORY_BYTES,
        max_depth: int = DEFAULT_JSON_MAX_DEPTH,
        max_scalar_bytes: int = DEFAULT_JSON_MAX_SCALAR_BYTES,
        max_estimated_bytes: int = DEFAULT_JSON_MAX_ESTIMATED_BYTES,
    ) -> None:
        if raw_memory_multiplier <= 0 or token_memory_bytes <= 0:
            raise ValueError("JSON memory weights must be positive")
        if max_depth <= 0 or max_scalar_bytes <= 0 or max_estimated_bytes <= 0:
            raise ValueError("JSON complexity limits must be positive")
        self.raw_memory_multiplier = int(raw_memory_multiplier)
        self.token_memory_bytes = int(token_memory_bytes)
        self.max_depth = int(max_depth)
        self.max_scalar_bytes = int(max_scalar_bytes)
        self.max_estimated_bytes = int(max_estimated_bytes)

        self.raw_bytes = 0
        self.tokens = 0
        self.depth = 0
        self.peak_depth = 0
        self._in_string = False
        self._escaped = False
        self._scalar_active = False
        self._scalar_bytes = 0
        self._native_enabled = _USE_NATIVE_JSON_MEMORY and all(
            value <= _NATIVE_INTEGER_MAX
            for value in (
                self.raw_memory_multiplier,
                self.token_memory_bytes,
                self.max_depth,
                self.max_scalar_bytes,
                self.max_estimated_bytes,
            )
        )

    @property
    def estimated_bytes(self) -> int:
        return (
            self.raw_bytes * self.raw_memory_multiplier
            + self.tokens * self.token_memory_bytes
        )

    def snapshot(self) -> JSONMemorySnapshot:
        return JSONMemorySnapshot(
            raw_bytes=self.raw_bytes,
            tokens=self.tokens,
            depth=self.depth,
            peak_depth=self.peak_depth,
            scalar_bytes=self._scalar_bytes,
            estimated_bytes=self.estimated_bytes,
            raw_memory_multiplier=self.raw_memory_multiplier,
            structural_item_memory_bytes=self.token_memory_bytes,
        )

    def feed(self, chunk: bytes | bytearray | memoryview) -> int:
        if self._native_enabled:
            return self._feed_native(chunk)
        return self._feed_python(chunk)

    def _feed_native(self, chunk: bytes | bytearray | memoryview) -> int:
        view = memoryview(chunk).cast("B")
        native_chunk = chunk if isinstance(chunk, bytes) else view.tobytes()
        try:
            result = _uni_api_native.scan_json_memory_chunk(
                native_chunk,
                self.raw_memory_multiplier,
                self.token_memory_bytes,
                self.max_depth,
                self.max_scalar_bytes,
                self.max_estimated_bytes,
                self.raw_bytes,
                self.tokens,
                self.depth,
                self.peak_depth,
                self._in_string,
                self._escaped,
                self._scalar_active,
                self._scalar_bytes,
            )
        except OverflowError:
            self._native_enabled = False
            return self._feed_python(chunk)

        (
            error_code,
            trigger_phase_code,
            self.raw_bytes,
            self.tokens,
            self.depth,
            self.peak_depth,
            self._in_string,
            self._escaped,
            self._scalar_active,
            self._scalar_bytes,
            estimated_bytes,
        ) = result
        if error_code == 0:
            return int(estimated_bytes)

        trigger_phases = {
            1: JSONMemoryComplexityTriggerPhase.CHUNK_RAW_CHARGE,
            2: JSONMemoryComplexityTriggerPhase.STRUCTURAL_ITEM_SCAN,
            3: JSONMemoryComplexityTriggerPhase.DEPTH_SCAN,
            4: JSONMemoryComplexityTriggerPhase.SCALAR_SCAN,
        }
        trigger_phase = trigger_phases.get(int(trigger_phase_code))
        if trigger_phase is None:
            raise RuntimeError(
                f"native JSON guard returned phase {trigger_phase_code}"
            )
        if error_code == 1:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_DEPTH,
                trigger_phase=trigger_phase,
                message=f"JSON nesting exceeds {self.max_depth} levels",
            )
        if error_code == 2:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_SCALAR_BYTES,
                trigger_phase=trigger_phase,
                message=f"JSON scalar exceeds {self.max_scalar_bytes} bytes",
            )
        if error_code == 3:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES,
                trigger_phase=trigger_phase,
                message=(
                    "JSON materialization estimate exceeds "
                    f"{self.max_estimated_bytes} bytes"
                ),
            )
        raise RuntimeError(f"native JSON guard returned error {error_code}")

    def _feed_python(self, chunk: bytes | bytearray | memoryview) -> int:
        view = memoryview(chunk).cast("B")
        scan_buffer = chunk if isinstance(chunk, (bytes, bytearray)) else view
        self.raw_bytes += len(view)
        if self.estimated_bytes > self.max_estimated_bytes:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES,
                trigger_phase=(
                    JSONMemoryComplexityTriggerPhase.CHUNK_RAW_CHARGE
                ),
                message=(
                    "JSON materialization estimate exceeds "
                    f"{self.max_estimated_bytes} bytes"
                ),
            )

        position = 0
        length = len(view)
        while position < length:
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                    position += 1
                    continue

                # Prompt bodies are dominated by long JSON strings. Let the
                # C regex engine skip directly to the next quote or escape
                # instead of dispatching one Python loop iteration per byte.
                match = _JSON_STRING_SPECIAL.search(scan_buffer, position)
                if match is None:
                    break
                position = match.start()
                value = view[position]
                position += 1
                if value == 0x5C:  # backslash
                    if position < length:
                        # The escaped byte has no structural meaning, including
                        # when it is itself a quote or backslash.
                        position += 1
                    else:
                        self._escaped = True
                else:  # quote
                    self._in_string = False
                continue

            # Outside strings, only JSON structure and scalar delimiters need
            # per-token handling. A scalar run can be charged in one step,
            # while preserving the exact max_scalar_bytes rejection point.
            match = _JSON_OUTSIDE_SPECIAL.search(scan_buffer, position)
            run_end = length if match is None else match.start()
            if run_end > position:
                if not self._scalar_active:
                    self._scalar_active = True
                    self._scalar_bytes = 0
                    self._count_token()
                scalar_bytes = self._scalar_bytes + (run_end - position)
                if scalar_bytes > self.max_scalar_bytes:
                    self._scalar_bytes = self.max_scalar_bytes + 1
                    self._raise_complexity(
                        reason=JSONMemoryComplexityReason.MAX_SCALAR_BYTES,
                        trigger_phase=(
                            JSONMemoryComplexityTriggerPhase.SCALAR_SCAN
                        ),
                        message=(
                            f"JSON scalar exceeds {self.max_scalar_bytes} bytes"
                        ),
                    )
                self._scalar_bytes = scalar_bytes
                position = run_end
            if match is None:
                break

            value = view[position]
            position += 1
            if value == 0x22:  # quote starts a key or value string
                self._finish_scalar()
                self._count_token()
                self._in_string = True
                continue

            if value in (0x7B, 0x5B):  # { [
                self._finish_scalar()
                self._count_token()
                self.depth += 1
                self.peak_depth = max(self.peak_depth, self.depth)
                if self.depth > self.max_depth:
                    self._raise_complexity(
                        reason=JSONMemoryComplexityReason.MAX_DEPTH,
                        trigger_phase=JSONMemoryComplexityTriggerPhase.DEPTH_SCAN,
                        message=f"JSON nesting exceeds {self.max_depth} levels",
                    )
                continue

            if value in (0x7D, 0x5D):  # } ]
                self._finish_scalar()
                self.depth = max(0, self.depth - 1)
                continue

            if value in (0x20, 0x09, 0x0A, 0x0D, 0x2C, 0x3A):
                # JSON whitespace, comma, or colon terminates a scalar token.
                self._finish_scalar()
                continue

        return self.estimated_bytes

    def _count_token(self) -> None:
        self.tokens += 1
        if self.estimated_bytes > self.max_estimated_bytes:
            self._raise_complexity(
                reason=JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES,
                trigger_phase=(
                    JSONMemoryComplexityTriggerPhase.STRUCTURAL_ITEM_SCAN
                ),
                message=(
                    "JSON materialization estimate exceeds "
                    f"{self.max_estimated_bytes} bytes"
                ),
            )

    def _raise_complexity(
        self,
        *,
        reason: JSONMemoryComplexityReason,
        trigger_phase: JSONMemoryComplexityTriggerPhase,
        message: str,
    ) -> None:
        configured_limit = {
            JSONMemoryComplexityReason.MAX_DEPTH: self.max_depth,
            JSONMemoryComplexityReason.MAX_SCALAR_BYTES: self.max_scalar_bytes,
            JSONMemoryComplexityReason.MAX_ESTIMATED_BYTES: (
                self.max_estimated_bytes
            ),
        }[reason]
        raise JSONMemoryComplexityError(
            message,
            observation=JSONMemoryComplexityObservation(
                schema_version=1,
                reason=reason,
                trigger_phase=trigger_phase,
                raw_bytes=self.raw_bytes,
                structural_item_count=self.tokens,
                depth=self.depth,
                peak_depth=self.peak_depth,
                scalar_bytes=self._scalar_bytes,
                estimated_bytes=self.estimated_bytes,
                configured_limit=configured_limit,
                max_depth=self.max_depth,
                max_scalar_bytes=self.max_scalar_bytes,
                max_estimated_bytes=self.max_estimated_bytes,
                raw_memory_multiplier=self.raw_memory_multiplier,
                structural_item_memory_bytes=self.token_memory_bytes,
            ),
        )

    def _finish_scalar(self) -> None:
        self._scalar_active = False
        self._scalar_bytes = 0


def estimate_json_memory_bytes(
    payload: bytes | bytearray | memoryview,
    **limits: int,
) -> JSONMemorySnapshot:
    estimator = IncrementalJSONMemoryEstimator(**limits)
    estimator.feed(payload)
    return estimator.snapshot()


def estimate_json_text_memory_bytes(
    payload: str,
    **limits: int,
) -> JSONMemorySnapshot:
    """Scan text without allocating a second attacker-sized UTF-8 copy."""

    estimator = IncrementalJSONMemoryEstimator(**limits)
    for offset in range(0, len(payload), _TEXT_SCAN_CHUNK_CHARACTERS):
        estimator.feed(
            payload[offset : offset + _TEXT_SCAN_CHUNK_CHARACTERS].encode(
                "utf-8",
                errors="strict",
            )
        )
    return estimator.snapshot()
