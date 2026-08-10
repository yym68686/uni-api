import random

import pytest

from uni_api.admission.json_memory import (
    IncrementalJSONMemoryEstimator,
    JSONMemoryComplexityError,
    json_memory_native_available,
)


def _outcome(estimator, chunks):
    values = []
    try:
        for chunk in chunks:
            values.append(estimator.feed(chunk))
    except JSONMemoryComplexityError as exc:
        return ("error", str(exc), exc.observation, estimator.snapshot())
    return ("ok", values, estimator.snapshot())


@pytest.mark.skipif(
    not json_memory_native_available(),
    reason="native JSON guard is not built",
)
def test_native_scanner_matches_python_for_random_chunked_buffers():
    random_source = random.Random(20260811)

    for _ in range(1_000):
        payload = bytes(
            random_source.randrange(256)
            for _ in range(random_source.randrange(512))
        )
        offsets = sorted(
            {
                0,
                len(payload),
                *(
                    random_source.randrange(len(payload) + 1)
                    for _ in range(12)
                ),
            }
        )
        raw_chunks = [
            payload[start:end]
            for start, end in zip(offsets, offsets[1:])
        ]
        chunk_type = random_source.randrange(3)
        if chunk_type == 1:
            chunks = [bytearray(chunk) for chunk in raw_chunks]
        elif chunk_type == 2:
            chunks = [memoryview(chunk) for chunk in raw_chunks]
        else:
            chunks = raw_chunks
        limits = {
            "raw_memory_multiplier": random_source.randrange(1, 8),
            "token_memory_bytes": random_source.randrange(1, 2048),
            "max_depth": random_source.randrange(1, 32),
            "max_scalar_bytes": random_source.randrange(1, 128),
            "max_estimated_bytes": random_source.randrange(1, 256 * 1024),
        }

        native = IncrementalJSONMemoryEstimator(**limits)
        native._native_enabled = True
        reference = IncrementalJSONMemoryEstimator(**limits)
        reference._native_enabled = False

        assert _outcome(native, chunks) == _outcome(reference, chunks)


def test_native_integer_envelope_falls_back_to_python():
    estimator = IncrementalJSONMemoryEstimator(
        max_estimated_bytes=1 << 80,
    )

    assert estimator._native_enabled is False
    assert estimator.feed(b'{"value":1}') == estimator.estimated_bytes
