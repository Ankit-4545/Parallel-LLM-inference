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
                 log_name: str = "coordinator", ready_timeout: float = 180.0, num_stages: int | None = None):
        self.port = port
        self.worker_base_port = worker_base_port or port + 1
        self.extra = list(extra or [])
        self.addr = f"127.0.0.1:{port}"
        self.log_path = Path(LOG_DIR) / f"{log_name}.log"
        self.ready_timeout = ready_timeout
        if num_stages is None:
            num_stages = int(self.extra[self.extra.index("--num-stages") + 1]) if "--num-stages" in self.extra else 2
        self.num_stages = num_stages
        self.proc: subprocess.Popen | None = None
        self.starts = 0

    def start(self) -> "LocalCluster":
        os.makedirs(LOG_DIR, exist_ok=True)
        cmd = [sys.executable, "-m", "pllm.coordinator", "--port", str(self.port),
               "--worker-base-port", str(self.worker_base_port), *self.extra]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self._log = open(self.log_path, "a" if self.starts else "w")
        self.starts += 1
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
        self.reap_workers()

    def kill(self) -> float:
        """SIGKILL the coordinator (crash-stop). Its workers are left alone: attached ones exit by
        themselves, detached ones keep running. Returns the wall-clock time of the kill."""
        t = time.time()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        return t

    def restart(self) -> "LocalCluster":
        """Start a new coordinator incarnation with the same arguments (what systemd/launchd would do)."""
        return self.start()

    def worker_health(self, stage: int, timeout: float = 0.5):
        """Synchronous Health probe of a worker port (no coordinator epoch is claimed). None if down."""
        import grpc

        from pllm.proto import pipeline_pb2 as pb
        from pllm.proto import pipeline_pb2_grpc as pb_grpc

        with grpc.insecure_channel(self.worker_addr(stage)) as ch:
            try:
                return pb_grpc.StageServiceStub(ch).Health(pb.HealthRequest(), timeout=timeout)
            except grpc.RpcError:
                return None

    def reap_workers(self) -> None:
        """Kill any worker still listening on this cluster's ports (e.g. detached workers whose
        coordinator was SIGKILLed and never restarted), so later clusters can reuse the ports."""
        from pllm.supervisor import kill_pid

        for stage in range(self.num_stages):
            h = self.worker_health(stage, timeout=0.3)
            if h is not None and h.pid:
                kill_pid(h.pid, signal.SIGTERM, timeout=5.0)

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
