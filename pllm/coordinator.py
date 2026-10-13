"""Client-facing coordinator (objectives O3-O7).

    python -m pllm.coordinator                      # spawns 2 local workers, listens on :50050
    python -m pllm.coordinator --worker-addrs hostA:50051,hostB:50052   # externally managed workers

Responsibilities
----------------
* Admission control: a bounded admission queue in front of a bounded set of
  active requests. A full queue yields an explicit OVERLOADED response.
* Decode loop: every active request alternates between "ready" and "in flight".
  The scheduler groups ready requests into micro-batches (max batch size, max
  wait time) and keeps up to `max_inflight_batches` batches in the pipeline,
  so stage 0 works on batch k+1 while stage 1 works on batch k.
* Ordering: each request has at most one step in flight; a result is only
  accepted if it matches the request's (in-flight batch, attempt, next step),
  so tokens are emitted strictly in step order and never twice.
* Failure handling: heartbeats detect dead workers; the coordinator restarts
  them with a higher epoch, discards every response produced under an old
  epoch, and rebuilds KV caches by replaying prompt + generated tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import signal
import time
import uuid
from collections import deque

import grpc

from pllm.config import GRPC_OPTIONS, CoordinatorConfig
from pllm.metrics import Counter
from pllm.proto import pipeline_pb2 as pb
from pllm.proto import pipeline_pb2_grpc as pb_grpc
from pllm.supervisor import WorkerSpec, WorkerSupervisor

log = logging.getLogger("pllm.coordinator")

# Item statuses that mean "this worker has lost or never had the request's cache": replay it.
_NEEDS_REPLAY = {pb.UNKNOWN_REQUEST, pb.REQUEST_CLOSED, pb.STALE_ATTEMPT, pb.STALE_STEP, pb.OUT_OF_ORDER}
# Transient statuses: retry the same step after a short backoff.
_TRANSIENT = {pb.QUEUE_FULL, pb.CACHE_FULL, pb.DOWNSTREAM_UNAVAILABLE, pb.INTERNAL_ERROR,
              pb.STALE_EPOCH, pb.FUTURE_EPOCH}
# Connectivity problems: a worker is unreachable or being replaced. Deciding that a worker is dead
# is the heartbeat monitor's job (and the recovery deadline bounds the wait), so these back off
# but do not consume the per-request retry budget.
_CONNECTIVITY = {pb.DOWNSTREAM_UNAVAILABLE, pb.STALE_EPOCH, pb.FUTURE_EPOCH}


class OverloadedError(Exception):
    pass


class Request:
    def __init__(self, client_id: str, prompt_ids: list[int], max_new_tokens: int, timeout_s: float):
        self.id = uuid.uuid4().hex          # internal id used on the wire; never reused
        self.client_id = client_id or self.id
        self.prompt = list(prompt_ids)
        self.max_new = max_new_tokens
        self.generated: list[int] = []
        self.next_step = 0
        self.attempt = 0
        self.needs_reset = True             # the first step is a prefill
        self.in_flight_batch: int | None = None
        self.done = False
        self.status = pb.RUNNING
        self.detail = ""
        self.events: asyncio.Queue = asyncio.Queue()
        self.t_submit = time.monotonic()
        self.t_admit: float | None = None
        self.t_first: float | None = None
        self.token_times: list[float] = []
        self.deadline = self.t_submit + timeout_s
        self.retries = 0                    # per-item errors; bounded by max_retries
        self.conn_failures = 0              # consecutive connectivity failures; only drives backoff
        self.recovering_since: float | None = None
        self.bytes_sent = 0                 # wire bytes attributed to this request (all hops)

    def mark_reset(self) -> None:
        """Schedule a cache rebuild (replay prompt + generated tokens) under a new attempt number."""
        self.needs_reset = True
        self.attempt += 1


class Coordinator:
    def __init__(self, cfg: CoordinatorConfig, tokenizer=None, n_positions: int = 1024):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.n_positions = n_positions
        n = cfg.num_stages
        self.supervisor: WorkerSupervisor | None = None
        if cfg.spawn_workers:
            self.supervisor = WorkerSupervisor(WorkerSpec(
                num_stages=n, split=cfg.split, base_port=cfg.worker_base_port, threads=cfg.worker_threads,
                queue_size=cfg.worker_queue_size, max_cached_requests=cfg.worker_max_cached_requests,
                activation_dtype=cfg.activation_dtype))
            self.addrs = self.supervisor.addrs()
        else:
            if len(cfg.worker_addrs) != n:
                raise ValueError(f"need {n} worker addresses, got {cfg.worker_addrs}")
            self.addrs = list(cfg.worker_addrs)
        self.epochs = [0] * n
        self.channels: list[grpc.aio.Channel] = []
        self.stubs: list[pb_grpc.StageServiceStub] = []

        self.admission: deque[Request] = deque()
        self.active: dict[str, Request] = {}
        self.by_client: dict[str, Request] = {}
        self.ready: deque[Request] = deque()
        self.inflight = 0
        self._batch_ids = itertools.count(1)
        self._wake = asyncio.Event()
        self._recovering: set[int] = set()
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self.t0 = time.monotonic()
        self.last_heartbeat = [time.monotonic()] * n
        self.recovery_events: list[dict] = []
        self._open_recovery: dict | None = None
        self.counters = Counter()
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.counters.clear()
        self.max_admission_depth = 0
        self.max_active = 0
        self.max_inflight = 0
        self.batch_sizes: list[int] = []
        self.batch_log: deque[dict] = deque(maxlen=20000)
        self.metrics_t0 = time.monotonic()

    # ------------------------------------------------------------------ lifecycle
    async def start(self, wait_ready_s: float = 120.0) -> None:
        if self.supervisor is not None:
            self.epochs = [1] * self.cfg.num_stages
            self.supervisor.start_all(epoch=1)
        for a in self.addrs:
            ch = grpc.aio.insecure_channel(a, options=GRPC_OPTIONS)
            self.channels.append(ch)
            self.stubs.append(pb_grpc.StageServiceStub(ch))
        deadline = time.monotonic() + wait_ready_s
        for s in range(self.cfg.num_stages):
            epoch = await self._wait_healthy(s, self.epochs[s] if self.supervisor else None, deadline)
            if epoch is None:
                raise RuntimeError(f"stage {s} at {self.addrs[s]} did not become healthy")
            self.epochs[s] = epoch
        self.last_heartbeat = [time.monotonic()] * self.cfg.num_stages
        self._running = True
        self._tasks = [asyncio.create_task(self._scheduler_loop(), name="scheduler"),
                       asyncio.create_task(self._health_loop(), name="health"),
                       asyncio.create_task(self._reaper_loop(), name="reaper")]
        log.info("coordinator ready: stages=%s epochs=%s", self.addrs, self.epochs)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        for r in list(self.active.values()) + list(self.admission):
            self._finish(r, pb.CANCELLED, "coordinator shutting down", release=False)
        for ch in self.channels:
            await ch.close()
        if self.supervisor is not None:
            await asyncio.get_running_loop().run_in_executor(None, self.supervisor.stop_all)

    def now(self) -> float:
        return time.monotonic() - self.t0

    # ------------------------------------------------------------------ admission
    def submit(self, prompt_ids: list[int], max_new_tokens: int, client_id: str = "",
               timeout_s: float = 0.0) -> Request:
        if not self._running:
            raise OverloadedError("coordinator not running")
        if len(self.admission) >= self.cfg.admission_queue_size:
            self.counters["rejected_overloaded"] += 1
            raise OverloadedError(f"admission queue full ({self.cfg.admission_queue_size})")
        r = Request(client_id, prompt_ids, max_new_tokens, timeout_s or self.cfg.request_timeout_s)
        self.admission.append(r)
        self.by_client[r.client_id] = r
        self.counters["submitted"] += 1
        self.max_admission_depth = max(self.max_admission_depth, len(self.admission))
        self._wake.set()
        return r

    def _admit(self) -> None:
        while self.admission and len(self.active) < self.cfg.max_active:
            r = self.admission.popleft()
            if r.done:
                continue
            r.t_admit = time.monotonic()
            self.active[r.id] = r
            self.ready.append(r)
            self.counters["admitted"] += 1
        self.max_active = max(self.max_active, len(self.active))

    async def cancel(self, client_id: str, reason: str = "cancelled by client") -> bool:
        r = self.by_client.get(client_id)
        if r is None or r.done:
            return False
        self._finish(r, pb.CANCELLED, reason)
        self.counters["cancelled"] += 1
        return True

    def _finish(self, r: Request, status: int, detail: str = "", release: bool = True) -> None:
        if r.done:
            return
        r.done = True
        r.status = status
        r.detail = detail
        self.active.pop(r.id, None)
        if self.by_client.get(r.client_id) is r:
            del self.by_client[r.client_id]
        try:
            self.admission.remove(r)
        except ValueError:
            pass
        now = time.monotonic()
        r.events.put_nowait(pb.GenerateChunk(
            request_id=r.client_id, status=status, final=True, detail=detail,
            ttft_ms=((r.t_first - (r.t_admit or r.t_submit)) * 1e3) if r.t_first else 0.0,
            total_ms=(now - r.t_submit) * 1e3, token_ids=r.generated))
        self.counters[f"finished_{pb.RequestStatus.Name(status).lower()}"] += 1
        # Free KV caches if the workers may still hold state for this request.
        if release and status != pb.COMPLETED and (r.next_step > 0 or r.in_flight_batch is not None):
            asyncio.get_running_loop().create_task(self._release_everywhere([r.id]))
        self._wake.set()

    async def _release_everywhere(self, ids: list[str]) -> None:
        for stub in self.stubs:
            try:
                await stub.Release(pb.ReleaseRequest(request_ids=ids), timeout=self.cfg.rpc_timeout_s)
            except grpc.aio.AioRpcError:
                pass  # a dead worker has no cache to free; TTL eviction covers the rest

    # ------------------------------------------------------------------ scheduling
    def _make_ready(self, r: Request) -> None:
        if not r.done and r.in_flight_batch is None and r.id in self.active:
            self.ready.append(r)
            self._wake.set()

    async def _wait(self, timeout: float | None = None) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    async def _scheduler_loop(self) -> None:
        cfg = self.cfg
        oldest_ready_at: float | None = None
        while self._running:
            self._admit()
            # drop finished/cancelled requests from the ready queue
            while self.ready and (self.ready[0].done or self.ready[0].in_flight_batch is not None):
                self.ready.popleft()
            if self._recovering or not self.ready or self.inflight >= cfg.max_inflight_batches:
                oldest_ready_at = None
                await self._wait(0.05)
                continue
            n_ready = sum(1 for r in self.ready if not r.done and r.in_flight_batch is None)
            if n_ready == 0:
                self.ready.clear()
                continue
            if oldest_ready_at is None:
                oldest_ready_at = time.monotonic()
            waited = time.monotonic() - oldest_ready_at
            # Wait (up to max_wait_ms) for a fuller batch only if more requests can still
            # become ready, i.e. some active requests are in flight or waiting for admission.
            more_coming = len(self.active) > n_ready or bool(self.admission)
            if n_ready < cfg.max_batch_size and more_coming and waited * 1e3 < cfg.max_wait_ms:
                await self._wait(cfg.max_wait_ms / 1e3 - waited)
                continue
            batch: list[Request] = []
            while self.ready and len(batch) < cfg.max_batch_size:
                r = self.ready.popleft()
                if not r.done and r.in_flight_batch is None and r not in batch:
                    batch.append(r)
            oldest_ready_at = None
            if batch:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
                bid = next(self._batch_ids)
                for r in batch:
                    r.in_flight_batch = bid
                asyncio.create_task(self._run_batch(bid, batch))

    def _build_item(self, r: Request) -> pb.StageItem:
        use_cache = self.cfg.kv_cache
        full = r.needs_reset or not use_cache
        tokens = r.prompt + r.generated if full else [r.generated[-1]]
        return pb.StageItem(request_id=r.id, step=r.next_step, attempt=r.attempt,
                            reset=use_cache and r.needs_reset, use_cache=use_cache,
                            release_after=use_cache and r.next_step == r.max_new - 1, token_ids=tokens)

    async def _run_batch(self, bid: int, batch: list[Request]) -> None:
        epochs = list(self.epochs)
        items = [self._build_item(r) for r in batch]
        msg = pb.ForwardRequest(stage_epochs=epochs, batch_id=bid, items=items)
        t0 = time.monotonic()
        resp, err = None, None
        try:
            resp = await self.stubs[0].Forward(msg, timeout=self.cfg.rpc_timeout_s)
        except grpc.aio.AioRpcError as e:
            err = e
        except Exception as e:  # pragma: no cover - defensive
            err = e
        finally:
            self.inflight -= 1
        t1 = time.monotonic()
        try:
            self._process_response(bid, epochs, batch, items, msg, resp, err, t0, t1)
        finally:
            self._wake.set()
        if self.cfg.duplicate_dispatch and resp is not None:
            asyncio.create_task(self._send_duplicate(msg, epochs, batch))

    async def _send_duplicate(self, msg: pb.ForwardRequest, epochs, batch) -> None:
        """Fault injection (E8): deliver the same micro-batch a second time."""
        self.counters["duplicates_sent"] += 1
        try:
            resp = await self.stubs[0].Forward(msg, timeout=self.cfg.rpc_timeout_s)
        except grpc.aio.AioRpcError:
            return
        # Process with a batch id no request is waiting on: every result must be discarded.
        self._process_response(-msg.batch_id, epochs, batch, list(msg.items), msg, resp, None,
                               time.monotonic(), time.monotonic(), duplicate=True)

    def _process_response(self, bid, epochs, batch, items, msg, resp, err, t0, t1, duplicate=False) -> None:
        cfg = self.cfg
        stale_epoch = epochs != self.epochs
        if not duplicate:
            self._record_batch(bid, items, msg, resp, t0, t1)
        if err is not None or stale_epoch:
            if stale_epoch:
                self.counters["discarded_stale_epoch_batches"] += 1
                if resp is not None:
                    self.counters["discarded_stale_epoch_results"] += len(resp.results)
            else:
                code = err.code().name if isinstance(err, grpc.aio.AioRpcError) else type(err).__name__
                self.counters[f"rpc_error_{code}"] += 1
            for r in batch:
                if r.in_flight_batch == bid:
                    r.in_flight_batch = None
                    if not r.done:
                        self._retry_after_connectivity_failure(r)
            return

        by_id = {r.id: r for r in batch}
        seen = set()
        for res in resp.results:
            r = by_id.get(res.request_id)
            if r is None or r.done or r.in_flight_batch != bid or duplicate:
                self.counters["discarded_duplicate_results"] += 1
                continue
            if res.step != r.next_step or res.attempt != r.attempt:
                self.counters["discarded_mismatched_results"] += 1
                continue
            seen.add(r.id)
            r.in_flight_batch = None
            if res.status == pb.OK:
                if res.replayed:
                    self.counters["idempotent_replays"] += 1
                self._accept(r, res.token_id)
                continue
            name = pb.ItemStatus.Name(res.status)
            self.counters[f"item_{name}"] += 1
            if res.status in _CONNECTIVITY:
                self._retry_after_connectivity_failure(r)
                continue
            r.retries += 1
            if r.retries > cfg.max_retries:
                self._finish(r, pb.FAILED, f"giving up after {r.retries} retries (last status {name})")
                continue
            if res.status in _NEEDS_REPLAY:
                r.mark_reset()
                self._make_ready(r)
            else:  # transient
                backoff = min(0.5, 0.01 * (2 ** min(r.retries, 6)))
                asyncio.get_running_loop().call_later(backoff, self._make_ready, r)
        for r in batch:  # items missing from the response (should not happen): retry
            if r.id not in seen and r.in_flight_batch == bid:
                r.in_flight_batch = None
                self._make_ready(r)

    def _retry_after_connectivity_failure(self, r: Request) -> None:
        """Retry the same step with exponential backoff (20 ms .. 250 ms). Not counted against
        max_retries: the request is bounded by the recovery deadline and its own timeout instead."""
        r.conn_failures += 1
        backoff = min(0.25, 0.01 * (2 ** min(r.conn_failures, 5)))
        asyncio.get_running_loop().call_later(backoff, self._make_ready, r)

    def _accept(self, r: Request, token: int) -> None:
        now = time.monotonic()
        r.conn_failures = 0
        if r.needs_reset:
            r.needs_reset = False
            if r.recovering_since is not None:
                r.recovering_since = None
                self._replay_done(r)
        step = r.next_step
        r.generated.append(token)
        r.next_step += 1
        if r.t_first is None:
            r.t_first = now
        r.token_times.append(now)
        self.counters["tokens_generated"] += 1
        text = self.tokenizer.decode([token]) if self.tokenizer is not None else ""
        r.events.put_nowait(pb.GenerateChunk(request_id=r.client_id, step=step, token_id=token, text=text,
                                             status=pb.RUNNING))
        if r.next_step >= r.max_new:
            self._finish(r, pb.COMPLETED)
        else:
            self._make_ready(r)

    def _record_batch(self, bid, items, msg, resp, t0, t1) -> None:
        self.batch_sizes.append(len(items))
        rec = {"t": t0 - self.t0, "batch_id": bid, "size": len(items), "rpc_ms": (t1 - t0) * 1e3,
               "tokens_in": sum(len(it.token_ids) for it in items),
               "prefill_items": sum(1 for it in items if it.reset or not it.use_cache),
               "bytes_c2s0": msg.ByteSize(), "ok": resp is not None}
        if resp is not None:
            for t in resp.timings:
                p = f"s{t.stage}_"
                rec.update({p + "queue_ms": t.queue_ms, p + "compute_ms": t.compute_ms,
                            p + "deser_ms": t.deserialize_ms, p + "ser_ms": t.serialize_ms,
                            p + "downstream_ms": t.downstream_ms, p + "bytes_out": t.bytes_out,
                            p + "link_delay_ms": t.link_delay_ms})
            # attribute wire bytes to requests in proportion to their token counts
            total_tokens = max(1, rec["tokens_in"])
            hop_bytes = rec["bytes_c2s0"] + sum(t.bytes_out for t in resp.timings) + resp.ByteSize()
            for it in items:
                r = self.active.get(it.request_id)
                if r is not None:
                    r.bytes_sent += hop_bytes * len(it.token_ids) / total_tokens
            self.counters["activation_bytes"] += sum(t.bytes_out for t in resp.timings)
            self.counters["wire_bytes"] += hop_bytes
        self.batch_log.append(rec)

    # ------------------------------------------------------------------ failure detection / recovery
    async def _ping(self, stage: int) -> pb.HealthResponse | None:
        try:
            return await self.stubs[stage].Health(pb.HealthRequest(), timeout=self.cfg.heartbeat_rpc_timeout_s)
        except grpc.aio.AioRpcError:
            return None

    async def _wait_healthy(self, stage: int, epoch: int | None, deadline: float) -> int | None:
        """Poll a stage until it answers (with `epoch`, if given). Returns the epoch or None."""
        while time.monotonic() < deadline:
            h = await self._ping(stage)
            if h is not None and h.ready and (epoch is None or h.epoch == epoch):
                return h.epoch
            await asyncio.sleep(0.05)
        return None

    async def _health_loop(self) -> None:
        cfg = self.cfg
        while self._running:
            replies = await asyncio.gather(*[self._ping(s) for s in range(cfg.num_stages)])
            now = time.monotonic()
            for s, h in enumerate(replies):
                if s in self._recovering:
                    continue
                if h is not None:
                    if h.epoch == self.epochs[s]:
                        self.last_heartbeat[s] = now
                    elif h.epoch > self.epochs[s] and self.supervisor is None:
                        # An externally restarted worker came back with a higher epoch: its caches are empty.
                        asyncio.create_task(self._recover(s, self.last_heartbeat[s], now, observed_epoch=h.epoch))
                        continue
                if now - self.last_heartbeat[s] > cfg.failure_timeout_s:
                    asyncio.create_task(self._recover(s, self.last_heartbeat[s], now))
            await asyncio.sleep(cfg.heartbeat_interval_s)

    async def _recover(self, stage: int, last_ok: float, detected: float, observed_epoch: int | None = None) -> None:
        if stage in self._recovering:
            return
        self._recovering.add(stage)
        cfg = self.cfg
        old_epoch = self.epochs[stage]
        ev = {"stage": stage, "old_epoch": old_epoch, "last_heartbeat_t": last_ok - self.t0,
              "detected_t": detected - self.t0, "affected_requests": len(self.active)}
        log.warning("stage %d declared dead (no heartbeat for %.0f ms); recovering",
                    stage, (detected - last_ok) * 1e3)
        self.counters["worker_failures"] += 1
        # Fence off the old incarnation: from now on every response computed under
        # the old epoch fails the epoch comparison and is discarded.
        self.epochs[stage] = -1
        for r in list(self.active.values()):
            r.mark_reset()
            r.retries = 0
            if r.recovering_since is None:
                r.recovering_since = detected
        deadline = detected + cfg.recovery_deadline_s
        new_epoch = None
        attempt = 0
        while time.monotonic() < deadline and new_epoch is None:
            attempt += 1
            if self.supervisor is not None:
                want = max(old_epoch, self.supervisor.epochs.get(stage, 0)) + 1
                t_restart = time.monotonic()
                await asyncio.get_running_loop().run_in_executor(None, self.supervisor.restart, stage, want)
                ev.setdefault("restart_issued_t", t_restart - self.t0)
                new_epoch = await self._wait_healthy(stage, want, min(deadline, time.monotonic() + 60))
            elif observed_epoch is not None:
                new_epoch = observed_epoch
            else:
                # External mode: wait for an operator / external supervisor to bring it back.
                h = await self._ping(stage)
                if h is not None and h.epoch > old_epoch:
                    new_epoch = h.epoch
                else:
                    await asyncio.sleep(0.1)
        if new_epoch is None:
            ev["result"] = "unrecoverable"
            self.recovery_events.append(ev)
            for r in list(self.active.values()) + list(self.admission):
                self._finish(r, pb.FAILED, f"stage {stage} could not be recovered within "
                                           f"{cfg.recovery_deadline_s:.0f}s")
            self.epochs[stage] = old_epoch
            self.last_heartbeat[stage] = time.monotonic()
            self._recovering.discard(stage)
            self._wake.set()
            return
        self.epochs[stage] = new_epoch
        self.last_heartbeat[stage] = time.monotonic()
        ev.update({"new_epoch": new_epoch, "restarted_t": self.now(), "restart_attempts": attempt,
                   "pending_replays": {r.id for r in self.active.values() if r.recovering_since is not None}})
        if not ev["pending_replays"]:
            ev["replayed_t"] = ev["restarted_t"]
        self._finalise_event(ev)
        self.recovery_events.append(ev)
        self._open_recovery = ev
        self._recovering.discard(stage)
        log.warning("stage %d recovered with epoch %d; replaying %d requests",
                    stage, new_epoch, len(ev["pending_replays"]))
        self._wake.set()

    def _replay_done(self, r: Request) -> None:
        for ev in reversed(self.recovery_events):
            pend = ev.get("pending_replays")
            if pend and r.id in pend:
                pend.discard(r.id)
                if not pend:
                    ev["replayed_t"] = self.now()
                    self._finalise_event(ev)
                break

    @staticmethod
    def _finalise_event(ev: dict) -> None:
        if "replayed_t" in ev and "restarted_t" in ev:
            ev["detection_ms"] = (ev["detected_t"] - ev["last_heartbeat_t"]) * 1e3
            ev["restart_ms"] = (ev["restarted_t"] - ev["detected_t"]) * 1e3
            ev["replay_ms"] = (ev["replayed_t"] - ev["restarted_t"]) * 1e3
            ev["recovery_ms"] = (ev["replayed_t"] - ev["last_heartbeat_t"]) * 1e3
            ev["result"] = "recovered"

    async def _reaper_loop(self) -> None:
        """Enforce request-level timeouts and the recovery deadline."""
        while self._running:
            now = time.monotonic()
            for r in list(self.active.values()) + list(self.admission):
                if now > r.deadline:
                    self._finish(r, pb.TIMEOUT, "request timeout")
                elif r.recovering_since is not None and not self._recovering \
                        and now - r.recovering_since > self.cfg.recovery_deadline_s:
                    self._finish(r, pb.FAILED, "could not recover request state before the deadline")
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ observability
    async def stats(self, reset: bool = False, include_workers: bool = False) -> dict:
        out = {
            "t": self.now(), "t0_monotonic": self.t0, "window_s": time.monotonic() - self.metrics_t0,
            "config": self.cfg.to_dict(), "epochs": self.epochs, "addrs": self.addrs,
            "admission_depth": len(self.admission), "admission_capacity": self.cfg.admission_queue_size,
            "active": len(self.active), "active_capacity": self.cfg.max_active,
            "ready": len(self.ready), "inflight_batches": self.inflight,
            "max_admission_depth": self.max_admission_depth, "max_active_seen": self.max_active,
            "max_inflight_seen": self.max_inflight, "recovering": sorted(self._recovering),
            "counters": dict(self.counters),
            "mean_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0.0,
            "batches": len(self.batch_sizes),
            "recovery_events": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in ev.items()
                                 if k != "pending_replays"} | {"pending_replays": len(ev.get("pending_replays", ()))}
                                for ev in self.recovery_events],
            "rss_bytes": _rss_bytes(),
            "worker_pids": ({s: self.supervisor.pid(s) for s in range(self.cfg.num_stages)}
                            if self.supervisor else {}),
        }
        if include_workers:
            out["workers"] = []
            for s, stub in enumerate(self.stubs):
                try:
                    resp = await stub.Stats(pb.JsonRequest(json=json.dumps({"reset": reset})), timeout=2.0)
                    out["workers"].append(json.loads(resp.json))
                except grpc.aio.AioRpcError as e:
                    out["workers"].append({"stage": s, "error": e.code().name})
        if reset:
            out["batch_log"] = list(self.batch_log)
            self._reset_metrics()
        return out


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        return -1


class CoordinatorServicer(pb_grpc.CoordinatorServiceServicer):
    def __init__(self, coord: Coordinator):
        self.c = coord

    async def Generate(self, request: pb.GenerateRequest, context):
        c = self.c
        ids = list(request.prompt_token_ids)
        if not ids and request.prompt:
            if c.tokenizer is None:
                yield pb.GenerateChunk(request_id=request.request_id, final=True, status=pb.INVALID,
                                       detail="no tokenizer available; send prompt_token_ids")
                return
            ids = c.tokenizer.encode(request.prompt).ids
        max_new = request.max_new_tokens or 32
        if not ids or max_new <= 0 or len(ids) + max_new > c.n_positions:
            yield pb.GenerateChunk(request_id=request.request_id, final=True, status=pb.INVALID,
                                   detail=f"need 1 <= prompt and prompt+max_new_tokens <= {c.n_positions}")
            return
        if request.request_id and request.request_id in c.by_client:
            yield pb.GenerateChunk(request_id=request.request_id, final=True, status=pb.INVALID,
                                   detail="request_id already in use")
            return
        try:
            r = c.submit(ids, max_new, request.request_id, request.timeout_s)
        except OverloadedError as e:
            yield pb.GenerateChunk(request_id=request.request_id, final=True, status=pb.OVERLOADED,
                                   detail=str(e))
            return
        try:
            while True:
                chunk = await r.events.get()
                yield chunk
                if chunk.final:
                    return
        finally:
            if not r.done:  # client went away / cancelled the stream
                await c.cancel(r.client_id, "client cancelled the stream")

    async def Cancel(self, request, context):
        return pb.CancelResponse(cancelled=await self.c.cancel(request.request_id))

    async def Stats(self, request, context):
        args = json.loads(request.json or "{}")
        s = await self.c.stats(reset=bool(args.get("reset")), include_workers=bool(args.get("workers")))
        return pb.JsonResponse(json=json.dumps(s, default=str))


def _load_tokenizer():
    try:
        from tokenizers import Tokenizer

        from pllm.model_loader import resolve_model_dir

        return Tokenizer.from_file(str(resolve_model_dir() / "tokenizer.json"))
    except Exception as e:  # pragma: no cover
        log.warning("tokenizer unavailable (%s); clients must send token ids", e)
        return None


async def serve(cfg: CoordinatorConfig) -> None:
    from pllm.model_loader import resolve_model_dir

    resolve_model_dir()  # download weights once, before workers start (they run offline)
    coord = Coordinator(cfg, tokenizer=_load_tokenizer())
    server = grpc.aio.server(options=GRPC_OPTIONS)
    pb_grpc.add_CoordinatorServiceServicer_to_server(CoordinatorServicer(coord), server)
    if server.add_insecure_port(f"{cfg.host}:{cfg.port}") == 0:
        raise RuntimeError(f"could not bind {cfg.host}:{cfg.port}")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await coord.start()
        await server.start()
        log.info("COORDINATOR READY on %s:%d (pid %d)", cfg.host, cfg.port, os.getpid())
        print(f"COORDINATOR READY {cfg.host}:{cfg.port}", flush=True)
        await stop.wait()
    finally:
        await server.stop(grace=0.5)
        await coord.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    d = CoordinatorConfig()
    ap = argparse.ArgumentParser(description="Pipeline coordinator")
    ap.add_argument("--port", type=int, default=d.port)
    ap.add_argument("--host", default=d.host)
    ap.add_argument("--num-stages", type=int, default=d.num_stages)
    ap.add_argument("--split", default=d.split, help='layer boundaries, e.g. "6", "4,8" or "auto"')
    ap.add_argument("--worker-addrs", default="", help="comma separated stage addresses; omit to spawn local workers")
    ap.add_argument("--worker-base-port", type=int, default=d.worker_base_port)
    ap.add_argument("--worker-threads", type=int, default=d.worker_threads)
    ap.add_argument("--worker-queue-size", type=int, default=d.worker_queue_size)
    ap.add_argument("--worker-max-cached", type=int, default=d.worker_max_cached_requests)
    ap.add_argument("--activation-dtype", default=d.activation_dtype, choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--admission-queue", type=int, default=d.admission_queue_size)
    ap.add_argument("--max-active", type=int, default=d.max_active)
    ap.add_argument("--no-kv-cache", action="store_true", help="disable the KV cache (recompute every step)")
    ap.add_argument("--max-batch-size", type=int, default=d.max_batch_size)
    ap.add_argument("--max-wait-ms", type=float, default=d.max_wait_ms)
    ap.add_argument("--max-inflight", type=int, default=d.max_inflight_batches,
                    help="micro-batches in the pipeline at once (1 disables pipelining)")
    ap.add_argument("--rpc-timeout", type=float, default=d.rpc_timeout_s)
    ap.add_argument("--request-timeout", type=float, default=d.request_timeout_s)
    ap.add_argument("--heartbeat-interval", type=float, default=d.heartbeat_interval_s)
    ap.add_argument("--failure-timeout", type=float, default=d.failure_timeout_s)
    ap.add_argument("--recovery-deadline", type=float, default=d.recovery_deadline_s)
    ap.add_argument("--duplicate-dispatch", action="store_true", help="fault injection: send every batch twice")
    ap.add_argument("--log-level", default="INFO")
    return ap


def config_from_args(a) -> CoordinatorConfig:
    addrs = [x for x in a.worker_addrs.split(",") if x]
    return CoordinatorConfig(
        port=a.port, host=a.host, num_stages=a.num_stages, split=a.split, worker_addrs=addrs,
        spawn_workers=not addrs, worker_base_port=a.worker_base_port, worker_threads=a.worker_threads,
        worker_queue_size=a.worker_queue_size, worker_max_cached_requests=a.worker_max_cached,
        activation_dtype=a.activation_dtype, admission_queue_size=a.admission_queue, max_active=a.max_active,
        kv_cache=not a.no_kv_cache, max_batch_size=a.max_batch_size, max_wait_ms=a.max_wait_ms,
        max_inflight_batches=a.max_inflight, rpc_timeout_s=a.rpc_timeout, request_timeout_s=a.request_timeout,
        heartbeat_interval_s=a.heartbeat_interval, failure_timeout_s=a.failure_timeout,
        recovery_deadline_s=a.recovery_deadline, duplicate_dispatch=a.duplicate_dispatch)


def main(argv=None) -> None:
    a = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s [coord] %(levelname)s %(message)s")
    asyncio.run(serve(config_from_args(a)))


if __name__ == "__main__":
    main()
