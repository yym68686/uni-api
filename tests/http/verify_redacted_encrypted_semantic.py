"""Redacted encrypted-content SSE uses the exact fallback, independent of unknown400."""
from pathlib import Path
import sys
from verify_encrypted_context_semantic import verify


if __name__ == '__main__':
    binary = Path(sys.argv[1]).resolve()
    for engine in ['gpt', 'codex']:
        for collect in [False, True]:
            verify(binary, engine, collect, redacted=True, unknown_retries=3)
    for unknown_retries in [0, 3]:
        verify(binary, 'codex', False, hedging=True, redacted=True,
               unknown_retries=unknown_retries)
