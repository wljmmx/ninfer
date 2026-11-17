"""Stop a real server with a live client still waiting for generation.

The property under test is the ordering, not the exit code by itself: the engine must be told to
stop BEFORE the HTTP worker queue is joined, so a handler blocked in an engine wait can return, the
response is written, and the process exits on its own. A server that joins first waits for the
request, which is the stall this test exists to catch.

POSIX sends SIGTERM. Windows has no SIGTERM for a child process: CTRL_BREAK_EVENT to a new process
group is the console equivalent (the server registers SIGBREAK for it), which is why the child is
started with CREATE_NEW_PROCESS_GROUP here.
"""

import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

IS_WINDOWS = os.name == "nt"


def main() -> int:
    artifact = os.environ.get("NINFER_TEST_ARTIFACT")
    if not artifact:
        print("SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact")
        return 77
    root = Path(__file__).resolve().parents[1] / ".temp/serve-shutdown-test"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        log_path = Path(temporary) / "server.log"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.argv[1], artifact, "--host", "127.0.0.1", "--port", str(port),
                 "--max-context", "8192", "--max-concurrency", "1", "--spec", "none",
                 "--no-cuda-graph", "--log-stats-interval-ms", "1000"],
                stdout=log, stderr=subprocess.STDOUT,
                # A new process group is what lets CTRL_BREAK_EVENT reach this child alone; POSIX
                # ignores it and the signal is sent directly below.
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
            )
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            try:
                endpoint = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise AssertionError("server exited during startup")
                    try:
                        with urllib.request.urlopen(endpoint + "/v1/models", timeout=1) as r:
                            model = json.load(r)["data"][0]["id"]
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise AssertionError("server startup timed out")
                        time.sleep(0.05)
                client.request("POST", "/v1/chat/completions", json.dumps({
                    "model": model, "stream": False, "max_tokens": 4096,
                    "messages": [{"role": "user", "content":
                        "Read this sequence: " + " item" * 3500 +
                        " Then write 2000 numbered sentences. Continue without summarizing."}],
                }), {"Content-Type": "application/json"})
                deadline = time.monotonic() + 60
                while True:
                    # A snapshot taken while the single lane is prefilling may miss its own 2s
                    # budget; that is not a failure, it is the same busy lane the fixture wants.
                    try:
                        with urllib.request.urlopen(endpoint + "/monitor/snapshot",
                                                    timeout=5) as r:
                            engine = json.load(r)["engine"]
                    except OSError:
                        engine = {}
                    # The counters live under engine.runtime; the per-request rows under
                    # engine.requests, which is what this fixture also pins: a live request must
                    # publish its state and its restoration fields, or an operator cannot tell a
                    # slow request from a rebuilding one.
                    runtime = engine.get("runtime", {})
                    rows = engine.get("requests", [])
                    busy = (runtime.get("running_requests", 0) +
                            runtime.get("prefilling_requests", 0))
                    if busy:
                        resident = [row for row in rows if row.get("resident")]
                        assert resident, "engine.requests published no resident row"
                        row = resident[0]
                        assert row["state"] in ("prefill", "decode", "control", "replay"), row
                        assert row["owner"] == "generation", row
                        assert set(row["recovery"]) == {
                            "route", "begin_frontier", "frontier", "target_frontier"}, row
                        assert row["last_progress_ns"] >= 0, row
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError("shutdown fixture never became resident")
                    time.sleep(0.01)
                started = time.monotonic()
                # Keep the client connected: a client disconnect must not rescue shutdown.
                if IS_WINDOWS:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGTERM)
                code = process.wait(timeout=30)
                elapsed = time.monotonic() - started
                assert code == 0, f"server exited with {code} after shutdown request"
                response = client.getresponse()
                response.read()
                assert response.status in (200, 503), (
                    f"in-flight request finished with {response.status}")
                print(f"PASS shutdown drained a resident request with the client connected in "
                      f"{elapsed:.3f}s (status {response.status})")
            except BaseException:
                print(log_path.read_text(errors="replace"), file=sys.stderr)
                raise
            finally:
                client.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
