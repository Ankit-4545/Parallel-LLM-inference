"""Pipeline worker process: serves one stage over gRPC (objective O3).

    python -m pllm.worker --stage 0 --num-stages 2 --port 50051 --next-addr 127.0.0.1:50052 --epoch 1
    python -m pllm.worker --stage 1 --num-stages 2 --port 50052 --epoch 1

Concurrency model
-----------------
* The gRPC server runs on an asyncio event loop. Health checks, stats and the
  forwarding of activations to the next stage all happen on the loop.
* All model compute (and all KV-cache mutation) happens on ONE dedicated
  compute thread. torch releases the GIL inside kernels, so the loop stays
  responsive to heartbeats while a batch is being computed.
* Because a Forward call releases the compute thread before it awaits the next
  stage, stage 0 can compute micro-batch k+1 while stage 1 computes batch k:
  this is what makes pipelining work.
* The number of Forward calls queued for / holding the compute thread is
  bounded by `queue_size`; beyond that the call is answered immediately with
  OVERLOADED (backpressure, never a silent drop).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch

from pllm.config import GRPC_OPTIONS, WorkerConfig
from pllm.model_loader import GPT2Config, resolve_model_dir, set_determinism
from pllm.partition import StageModule, stage_ranges
from pllm.proto import pipeline_pb2 as pb
from pllm.proto import pipeline_pb2_grpc as pb_grpc
from pllm.serialization import dtype_from_name, proto_to_tensor, tensor_to_proto
from pllm.stage_executor import StageExecutor

log = logging.getLogger("pllm.worker")


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        return -1


class StageServicer(pb_grpc.StageServiceServicer):
    def __init__(self, cfg: WorkerConfig, module: StageModule):
        self.cfg = cfg
        self.module = module
        self.executor = StageExecutor(module, cfg.stage, cfg.epoch, cfg.max_cached_requests, cfg.cache_ttl_s)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"stage{cfg.stage}-compute")
        self.wire_dtype = dtype_from_name(cfg.activation_dtype)
        self.link_delay_ms = cfg.link_delay_ms
        self.link_bandwidth_mbps = 0.0  # 0 = unlimited
        self.next_channel = None
        self.next_stub = None
        if cfg.next_addr:
            self.next_channel = grpc.aio.insecure_channel(cfg.next_addr, options=GRPC_OPTIONS)
            self.next_stub = pb_grpc.StageServiceStub(self.next_channel)
        self.pending = 0
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.started = time.monotonic()
        self.busy_s = 0.0
        self.forward_calls = 0
        self.max_pending = 0
        self.overloaded = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.activation_bytes_out = 0
        self.downstream_errors = 0
        self.executor.stats.__init__()

    # ------------------------------------------------------------------
    def _compute(self, request: pb.ForwardRequest, t_enqueued: float):
        """Runs on the compute thread: deserialise, validate+execute, serialise for the next stage."""
        t0 = time.perf_counter()
        queue_ms = (t0 - t_enqueued) * 1e3
        hidden_in = [None] * len(request.items)
        if not self.module.has_embeddings:
            for i, it in enumerate(request.items):
                if it.HasField("hidden"):
                    hidden_in[i] = proto_to_tensor(it.hidden, out_dtype=torch.float32)
        t1 = time.perf_counter()
        outcomes = self.executor.process(request.stage_epochs, request.items, hidden_in)
        t2 = time.perf_counter()
        next_req = None
        if self.next_stub is not None:
            items = []
            for it, o in zip(request.items, outcomes):
                if o.status == pb.OK:
                    items.append(pb.StageItem(
                        request_id=it.request_id, step=it.step, attempt=it.attempt, reset=it.reset,
                        use_cache=it.use_cache, release_after=it.release_after,
                        hidden=tensor_to_proto(o.hidden, self.wire_dtype)))
            if items:
                next_req = pb.ForwardRequest(stage_epochs=request.stage_epochs,
                                             batch_id=request.batch_id, items=items)
        t3 = time.perf_counter()
        self.busy_s += t3 - t0
        timing = pb.StageTiming(stage=self.cfg.stage, queue_ms=queue_ms, compute_ms=(t2 - t1) * 1e3,
                                deserialize_ms=(t1 - t0) * 1e3, serialize_ms=(t3 - t2) * 1e3)
        return outcomes, next_req, timing

    async def Forward(self, request: pb.ForwardRequest, context) -> pb.ForwardResponse:
        bytes_in = request.ByteSize()
        self.bytes_in += bytes_in
        self.forward_calls += 1
        if self.pending >= self.cfg.queue_size:
            self.overloaded += 1
            return pb.ForwardResponse(
                results=[pb.ItemResult(request_id=it.request_id, step=it.step, attempt=it.attempt,
                                       status=pb.QUEUE_FULL, detail=f"stage {self.cfg.stage} queue full")
                         for it in request.items],
                worker_epoch=self.cfg.epoch)
        self.pending += 1
        self.max_pending = max(self.max_pending, self.pending)
        loop = asyncio.get_running_loop()
        try:
            outcomes, next_req, timing = await loop.run_in_executor(
                self.pool, self._compute, request, time.perf_counter())
        finally:
            self.pending -= 1
        timing.bytes_in = bytes_in

        results = []
        for o in outcomes:
            r = pb.ItemResult(request_id=o.request_id, step=o.step, attempt=o.attempt, status=o.status,
                              replayed=o.replayed, detail=o.detail)
            if o.status == pb.OK and self.module.has_head:
                r.token_id = o.token_id
            results.append(r)
        timings = [timing]

        if next_req is not None:
            downstream = await self._forward_downstream(next_req, context, timing)
            by_id = {r.request_id: r for r in downstream.results} if downstream else {}
            for r in results:
                if r.status != pb.OK:
                    continue
                d = by_id.get(r.request_id)
                if d is None:
                    r.status = pb.DOWNSTREAM_UNAVAILABLE
                    r.detail = f"stage {self.cfg.stage + 1} unreachable"
                else:
                    replayed_here = r.replayed
                    r.CopyFrom(d)
                    r.replayed = r.replayed or replayed_here
            if downstream:
                timings.extend(downstream.timings)
        return pb.ForwardResponse(results=results, timings=timings, worker_epoch=self.cfg.epoch)

    async def _forward_downstream(self, next_req, context, timing) -> pb.ForwardResponse | None:
        nbytes = next_req.ByteSize()
        timing.bytes_out = nbytes
        self.bytes_out += nbytes
        self.activation_bytes_out += sum(len(it.hidden.data) for it in next_req.items)
        t0 = time.perf_counter()
        delay = self.link_delay_ms / 1e3
        if self.link_bandwidth_mbps > 0:
            delay += nbytes * 8 / (self.link_bandwidth_mbps * 1e6)
        if delay > 0:
            timing.link_delay_ms = delay * 1e3
            await asyncio.sleep(delay)
        timeout = self.cfg.downstream_timeout_s
        remaining = context.time_remaining()
        if remaining is not None:
            timeout = max(0.01, min(timeout, remaining - 0.01))
        try:
            return await self.next_stub.Forward(next_req, timeout=timeout)
        except grpc.aio.AioRpcError as e:
            self.downstream_errors += 1
            log.warning("downstream Forward failed: %s %s", e.code().name, e.details())
            return None
        finally:
            timing.downstream_ms = (time.perf_counter() - t0) * 1e3

    async def Health(self, request, context) -> pb.HealthResponse:
        return pb.HealthResponse(stage=self.cfg.stage, epoch=self.cfg.epoch, pid=os.getpid(),
                                 queue_depth=self.pending, cached_requests=len(self.executor.cache),
                                 ready=True)

    async def Release(self, request, context) -> pb.ReleaseResponse:
        loop = asyncio.get_running_loop()
        n = await loop.run_in_executor(self.pool, self.executor.release, list(request.request_ids))
        return pb.ReleaseResponse(released=n)

    async def Stats(self, request, context) -> pb.JsonResponse:
        args = json.loads(request.json or "{}")
        es = self.executor.stats
        wall = time.monotonic() - self.started
        out = {
            "stage": self.cfg.stage, "epoch": self.cfg.epoch, "pid": os.getpid(),
            "layers": [self.module.layer_start, self.module.layer_end],
            "param_bytes": self.module.param_bytes(), "rss_bytes": _rss_bytes(),
            "wall_s": wall, "busy_s": self.busy_s, "utilisation": self.busy_s / wall if wall > 0 else 0.0,
            "forward_calls": self.forward_calls, "batches": es.batches, "items": es.items,
            "tokens_processed": es.tokens_processed, "replayed": es.replayed, "rejected": es.rejected,
            "queue_depth": self.pending, "max_queue_depth": self.max_pending, "queue_size": self.cfg.queue_size,
            "overloaded": self.overloaded, "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
            "activation_bytes_out": self.activation_bytes_out, "downstream_errors": self.downstream_errors,
            "cached_requests": len(self.executor.cache), "cache_bytes": self.executor.cache.total_bytes(),
            "cache_evictions": self.executor.cache.evictions,
            "link_delay_ms": self.link_delay_ms, "activation_dtype": str(self.wire_dtype).replace("torch.", ""),
        }
        if args.get("reset"):
            self._reset_counters()
        return pb.JsonResponse(json=json.dumps(out))

    async def Configure(self, request, context) -> pb.JsonResponse:
        args = json.loads(request.json or "{}")
        if "link_delay_ms" in args:
            self.link_delay_ms = float(args["link_delay_ms"])
        if "link_bandwidth_mbps" in args:
            self.link_bandwidth_mbps = float(args["link_bandwidth_mbps"])
        if "activation_dtype" in args:
            self.wire_dtype = dtype_from_name(args["activation_dtype"])
        log.info("configured: %s", args)
        return pb.JsonResponse(json=json.dumps({"ok": True}))


async def serve(cfg: WorkerConfig) -> None:
    set_determinism(threads=cfg.threads)
    t0 = time.perf_counter()
    model_dir = resolve_model_dir(cfg.model_name, cfg.model_revision)
    gcfg = GPT2Config.from_json(model_dir / "config.json")
    start, end = stage_ranges(gcfg, cfg.num_stages, cfg.split)[cfg.stage]
    module = StageModule.from_pretrained(model_dir, start, end)
    log.info("stage %d epoch %d: layers [%d, %d) embeddings=%s head=%s params=%.1fM (%.1f MB) loaded in %.2fs",
             cfg.stage, cfg.epoch, start, end, module.has_embeddings, module.has_head,
             module.num_params() / 1e6, module.param_bytes() / 2**20, time.perf_counter() - t0)

    servicer = StageServicer(cfg, module)
    server = grpc.aio.server(options=GRPC_OPTIONS)
    pb_grpc.add_StageServiceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    if bound == 0:
        raise RuntimeError(f"could not bind {cfg.host}:{cfg.port}")
    await server.start()
    log.info("READY stage=%d epoch=%d listening on %s:%d next=%s pid=%d",
             cfg.stage, cfg.epoch, cfg.host, cfg.port, cfg.next_addr, os.getpid())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def watch_parent() -> None:
        # Exit if the supervising coordinator dies, so no orphaned workers are left behind.
        while not stop.is_set():
            if os.getppid() != cfg.parent_pid:
                log.warning("parent %d exited; shutting down", cfg.parent_pid)
                stop.set()
            await asyncio.sleep(1.0)

    if cfg.parent_pid:
        asyncio.create_task(watch_parent())
    await stop.wait()
    await server.stop(grace=0.5)
    if servicer.next_channel is not None:
        await servicer.next_channel.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pipeline stage worker")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--split", default="6", help='layer boundaries, e.g. "6", "4,8" or "auto"')
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to accept remote stages)")
    ap.add_argument("--next-addr", default=None, help="host:port of the next stage")
    ap.add_argument("--epoch", type=int, default=None,
                    help="incarnation number; defaults to the current time in ms so restarts always increase it")
    ap.add_argument("--threads", type=int, default=WorkerConfig.threads)
    ap.add_argument("--queue-size", type=int, default=WorkerConfig.queue_size)
    ap.add_argument("--max-cached-requests", type=int, default=WorkerConfig.max_cached_requests)
    ap.add_argument("--cache-ttl", type=float, default=WorkerConfig.cache_ttl_s)
    ap.add_argument("--activation-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--link-delay-ms", type=float, default=0.0)
    ap.add_argument("--model", default=WorkerConfig.model_name)
    ap.add_argument("--parent-pid", type=int, default=0, help="exit when this process exits (set by the supervisor)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format=f"%(asctime)s [stage{a.stage}] %(levelname)s %(message)s")
    cfg = WorkerConfig(stage=a.stage, num_stages=a.num_stages, port=a.port, split=a.split,
                       next_addr=a.next_addr, epoch=a.epoch if a.epoch is not None else int(time.time() * 1000),
                       host=a.host, model_name=a.model, threads=a.threads, queue_size=a.queue_size,
                       max_cached_requests=a.max_cached_requests, cache_ttl_s=a.cache_ttl,
                       activation_dtype=a.activation_dtype, link_delay_ms=a.link_delay_ms,
                       parent_pid=a.parent_pid)
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
