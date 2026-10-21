"""Loopback PostgreSQL-ingest protocol fixture; never connects to real storage."""
import hashlib
import json
import threading

_lock = threading.Lock()


def accept_facts(handler, destination):
    assert handler.headers.get("Authorization") == "Bearer fixture"
    raw = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
    events = [json.loads(line) for line in raw.splitlines() if line]
    with _lock:
        known = {event["event_id"] for event in destination}
        for event in events:
            if event["event_id"] not in known:
                destination.append(event)
                known.add(event["event_id"])
    response = json.dumps({"format": "facts-ack-v1", "sha256": hashlib.sha256(raw).hexdigest()}).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(response)))
    handler.end_headers()
    handler.wfile.write(response)
