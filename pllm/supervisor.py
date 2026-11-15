"""Spawns, kills and restarts local worker processes on behalf of the coordinator.

Two lifetimes:

* attached (default): workers get --parent-pid and exit when the coordinator dies;
* detached: workers run in their own session with an orphan timeout, survive a
  coordinator crash, and a restarted coordinator *adopts* them (`adopt`) instead
  of paying the weight-loading cost of a new process.
"""

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
    detached: bool = False
    orphan_timeout_s: float = 300.0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie child still "exists"; reap it if it is ours.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def kill_pid(pid: int, sig: int = signal.SIGKILL, timeout: float = 5.0) -> None:
    """Signal a process that may not be our child and wait until it is gone."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class WorkerSupervisor:
    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.procs: dict[int, subprocess.Popen] = {}
        self.adopted: dict[int, int] = {}       # stage -> pid of a worker started by an earlier coordinator
        self.epochs: dict[int, int] = {}
        self.coordinator_epoch = 0              # passed to new workers so they start fenced
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
               "--activation-dtype", s.activation_dtype,
               "--coordinator-epoch", str(self.coordinator_epoch)]
        if s.detached:
            cmd += ["--orphan-timeout", str(s.orphan_timeout_s)]
        else:
            cmd += ["--parent-pid", str(os.getpid())]
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
        # A detached worker gets its own session, so a signal to the coordinator's process group
        # (e.g. Ctrl-C in the terminal) does not reach it.
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=s.detached)
        self.procs[stage] = p
        self.adopted.pop(stage, None)
        self.epochs[stage] = epoch
        return p

    def adopt(self, stage: int, pid: int, epoch: int) -> None:
        """Take over a running worker that an earlier coordinator incarnation started."""
        self.procs.pop(stage, None)
        self.adopted[stage] = pid
        self.epochs[stage] = epoch

    def start_all(self, epoch: int = 1) -> None:
        # Start the last stage first so earlier stages can connect to it immediately.
        for stage in reversed(range(self.spec.num_stages)):
            self.spawn(stage, epoch)

    def pid(self, stage: int) -> int | None:
        p = self.procs.get(stage)
        if p is not None:
            return p.pid
        return self.adopted.get(stage)

    def alive(self, stage: int) -> bool:
        p = self.procs.get(stage)
        if p is not None:
            return p.poll() is None
        pid = self.adopted.get(stage)
        return pid is not None and pid_alive(pid)

    def kill(self, stage: int, sig: int = signal.SIGKILL) -> None:
        p = self.procs.get(stage)
        if p is not None:
            if p.poll() is None:
                p.send_signal(sig)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            return
        pid = self.adopted.get(stage)
        if pid is not None:
            kill_pid(pid, sig)

    def restart(self, stage: int, epoch: int) -> subprocess.Popen:
        """Crash-stop the old incarnation (if still running) and start a new one with a higher epoch."""
        self.kill(stage)
        return self.spawn(stage, epoch)

    def stop_all(self, timeout: float = 5.0) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for pid in self.adopted.values():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for p in self.procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for pid in self.adopted.values():
            kill_pid(pid, signal.SIGTERM, timeout=max(0.1, deadline - time.monotonic()))
        for f in self._logs.values():
            f.close()
        self._logs.clear()
