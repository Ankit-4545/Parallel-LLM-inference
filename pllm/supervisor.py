"""Spawns, kills and restarts local worker processes on behalf of the coordinator."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from pllm.config import LOG_DIR, REPO_ROOT


@dataclass
class WorkerSpec:
    num_stages: int
    split: str = "6"
    host: str = "127.0.0.1"
    base_port: int = 50051
    threads: int = 4
    queue_size: int = 8
    max_cached_requests: int = 128
    activation_dtype: str = "float32"
    log_dir: str = str(LOG_DIR)


class WorkerSupervisor:
    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.procs: dict[int, subprocess.Popen] = {}
        self.epochs: dict[int, int] = {}
        self._logs: dict[int, object] = {}
        os.makedirs(spec.log_dir, exist_ok=True)

    def addr(self, stage: int) -> str:
        return f"{self.spec.host}:{self.spec.base_port + stage}"

    def addrs(self) -> list[str]:
        return [self.addr(s) for s in range(self.spec.num_stages)]

    def spawn(self, stage: int, epoch: int) -> subprocess.Popen:
        s = self.spec
        cmd = [sys.executable, "-m", "pllm.worker",
               "--stage", str(stage), "--num-stages", str(s.num_stages), "--split", s.split,
               "--port", str(s.base_port + stage), "--host", s.host, "--epoch", str(epoch),
               "--threads", str(s.threads), "--queue-size", str(s.queue_size),
               "--max-cached-requests", str(s.max_cached_requests),
               "--activation-dtype", s.activation_dtype, "--parent-pid", str(os.getpid())]
        if stage + 1 < s.num_stages:
            cmd += ["--next-addr", self.addr(stage + 1)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("HF_HUB_OFFLINE", "1")  # weights are already cached by the coordinator
        old = self._logs.pop(stage, None)
        if old is not None:
            old.close()
        logf = open(os.path.join(s.log_dir, f"stage{stage}_epoch{epoch}.log"), "w")
        self._logs[stage] = logf
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
        self.procs[stage] = p
        self.epochs[stage] = epoch
        return p

    def start_all(self, epoch: int = 1) -> None:
        # Start the last stage first so earlier stages can connect to it immediately.
        for stage in reversed(range(self.spec.num_stages)):
            self.spawn(stage, epoch)

    def pid(self, stage: int) -> int | None:
        p = self.procs.get(stage)
        return p.pid if p is not None else None

    def alive(self, stage: int) -> bool:
        p = self.procs.get(stage)
        return p is not None and p.poll() is None

    def kill(self, stage: int, sig: int = signal.SIGKILL) -> None:
        p = self.procs.get(stage)
        if p is not None and p.poll() is None:
            p.send_signal(sig)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def restart(self, stage: int, epoch: int) -> subprocess.Popen:
        """Crash-stop the old incarnation (if still running) and start a new one with a higher epoch."""
        self.kill(stage)
        return self.spawn(stage, epoch)

    def stop_all(self, timeout: float = 5.0) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + timeout
        for p in self.procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for f in self._logs.values():
            f.close()
        self._logs.clear()
