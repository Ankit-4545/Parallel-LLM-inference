"""Client library and CLI for the coordinator.

    python -m pllm.client --prompt "The capital of France is" --max-new-tokens 20
    python -m pllm.client --stats            # coordinator + worker statistics (JSON)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import grpc

from pllm.config import COORDINATOR_PORT, GRPC_OPTIONS
from pllm.proto import pipeline_pb2 as pb
from pllm.proto import pipeline_pb2_grpc as pb_grpc


@dataclass
class GenerateResult:
    request_id: str
    status: str
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    detail: str = ""
    t_start: float = 0.0
    token_times: list[float] = field(default_factory=list)  # client-side arrival time of each token
    t_end: float = 0.0
    server_ttft_ms: float = 0.0
    server_total_ms: float = 0.0
    final_token_ids: list[int] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        return (self.token_times[0] - self.t_start) * 1e3 if self.token_times else float("nan")

    @property
    def itl_ms(self) -> list[float]:
        return [(b - a) * 1e3 for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def latency_ms(self) -> float:
        return (self.t_end - self.t_start) * 1e3


def channel(addr: str) -> grpc.aio.Channel:
    return grpc.aio.insecure_channel(addr, options=GRPC_OPTIONS)


async def generate(stub: pb_grpc.CoordinatorServiceStub, *, prompt: str = "", prompt_ids=None,
                   max_new_tokens: int = 32, request_id: str = "", timeout_s: float = 0.0,
                   on_token=None, cancel_after: int | None = None, rpc_timeout: float | None = None
                   ) -> GenerateResult:
    req = pb.GenerateRequest(request_id=request_id, prompt=prompt, prompt_token_ids=list(prompt_ids or []),
                             max_new_tokens=max_new_tokens, timeout_s=timeout_s)
    res = GenerateResult(request_id=request_id, status="RUNNING", t_start=time.monotonic())
    call = stub.Generate(req, timeout=rpc_timeout)
    try:
        async for chunk in call:
            if chunk.final:
                res.status = pb.RequestStatus.Name(chunk.status)
                res.detail = chunk.detail
                res.server_ttft_ms = chunk.ttft_ms
                res.server_total_ms = chunk.total_ms
                res.final_token_ids = list(chunk.token_ids)
                break
            res.tokens.append(chunk.token_id)
            res.token_times.append(time.monotonic())
            res.text += chunk.text
            if on_token is not None:
                on_token(chunk)
            if cancel_after is not None and len(res.tokens) >= cancel_after:
                call.cancel()
                res.status = "CANCELLED"
                break
    except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.CANCELLED:
            res.status = "CANCELLED"
        else:
            res.status = "RPC_ERROR"
            res.detail = f"{e.code().name}: {e.details()}"
    except asyncio.CancelledError:
        call.cancel()
        raise
    res.t_end = time.monotonic()
    return res


async def get_stats(addr: str, reset: bool = False, workers: bool = True) -> dict:
    async with channel(addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)
        resp = await stub.Stats(pb.JsonRequest(json=json.dumps({"reset": reset, "workers": workers})), timeout=10)
        return json.loads(resp.json)


async def _main(a) -> int:
    if a.stats:
        print(json.dumps(await get_stats(a.addr, reset=a.reset_stats), indent=2, default=str))
        return 0
    async with channel(a.addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)
        if a.prompt is None:
            print("--prompt is required", file=sys.stderr)
            return 2
        print(a.prompt, end="", flush=True)
        res = await generate(stub, prompt=a.prompt, max_new_tokens=a.max_new_tokens, request_id=a.request_id,
                             timeout_s=a.timeout, cancel_after=a.cancel_after,
                             on_token=lambda c: print(c.text, end="", flush=True))
        print()
        print(f"[status={res.status} tokens={len(res.tokens)} ttft={res.ttft_ms:.1f}ms "
              f"total={res.latency_ms:.1f}ms{' detail=' + res.detail if res.detail else ''}]", file=sys.stderr)
        return 0 if res.status in ("COMPLETED", "CANCELLED") else 1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Coordinator client")
    ap.add_argument("--addr", default=f"127.0.0.1:{COORDINATOR_PORT}")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--request-id", default="")
    ap.add_argument("--timeout", type=float, default=0.0, help="request timeout in seconds (0 = server default)")
    ap.add_argument("--cancel-after", type=int, default=None, help="cancel the stream after N tokens")
    ap.add_argument("--stats", action="store_true", help="print coordinator and worker stats")
    ap.add_argument("--reset-stats", action="store_true")
    sys.exit(asyncio.run(_main(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
