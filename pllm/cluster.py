"""Launch a full local cluster (coordinator process + its worker processes) for tests,
benchmarks, the load test and the fault-injection script."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from pllm.config import LOG_DIR, REFERENCE_FILE, REPO_ROOT


class LocalCluster:
    """Context manager running `python -m pllm.coordinator <args>` as a child process.

    Example:
        with LocalCluster(port=50150, extra=["--max-batch-size", "4"]) as c:
            ... talk to c.addr ...
    """

    def __init__(self, port: int = 50150, worker_base_port: int | None = None, extra: list[str] | None = None,
                 log_name: str = "coordinator", ready_timeout: float = 180.0):
        self.port = port
        self.worker_base_port = worker_base_port or port + 1
        self.extra = list(extra or [])
        self.addr = f"127.0.0.1:{port}"
        self.log_path = Path(LOG_DIR) / f"{log_name}.log"
        self.ready_timeout = ready_timeout
        self.proc: subprocess.Popen | None = None

    def start(self) -> "LocalCluster":
        os.makedirs(LOG_DIR, exist_ok=True)
        cmd = [sys.executable, "-m", "pllm.coordinator", "--port", str(self.port),
               "--worker-base-port", str(self.worker_base_port), *self.extra]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        ready = threading.Event()

        def pump() -> None:
            for line in self.proc.stdout:
                self._log.write(line)
                self._log.flush()
                if "COORDINATOR READY" in line:
                    ready.set()

        threading.Thread(target=pump, daemon=True).start()
        deadline = time.monotonic() + self.ready_timeout
        while not ready.wait(0.1):
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"coordinator failed to start; see {self.log_path}")
        return self

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def __enter__(self) -> "LocalCluster":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- helpers ------------------------------------------------------------
    def stats(self, reset: bool = False, workers: bool = True) -> dict:
        from pllm.client import get_stats

        return asyncio.run(get_stats(self.addr, reset=reset, workers=workers))

    async def astats(self, reset: bool = False, workers: bool = True) -> dict:
        from pllm.client import get_stats

        return await get_stats(self.addr, reset=reset, workers=workers)

    def worker_pid(self, stage: int, stats: dict | None = None) -> int:
        stats = stats or self.stats(workers=False)
        return int(stats["worker_pids"][str(stage)])

    def worker_addr(self, stage: int) -> str:
        return f"127.0.0.1:{self.worker_base_port + stage}"


def load_reference(path=REFERENCE_FILE) -> dict:
    with open(path) as f:
        return json.load(f)
