"""Client library and CLI for the coordinator.

    python -m pllm.client --prompt "The capital of France is" --max-new-tokens 20
    python -m pllm.client --stats            # coordinator + worker statistics (JSON)
    python -m pllm.client --prompt "..." --request-id r1 --reconnect 60   # survive a coordinator restart
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
    reconnects: int = 0                                      # streams re-attached with Resume
    broken_at: list[float] = field(default_factory=list)    # when a stream broke (client clock)
    resumed_at: list[float] = field(default_factory=list)   # when a resumed stream delivered its first chunk

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


SUPERSEDED = "coordinator superseded"  # must match pllm.coordinator.SUPERSEDED


async def generate(stub: pb_grpc.CoordinatorServiceStub, *, prompt: str = "", prompt_ids=None,
                   max_new_tokens: int = 32, request_id: str = "", timeout_s: float = 0.0,
                   on_token=None, cancel_after: int | None = None, rpc_timeout: float | None = None,
                   reconnect_s: float = 0.0) -> GenerateResult:
    """Stream one generation. With reconnect_s > 0, a broken stream (coordinator crashed or was
    replaced) is re-attached with Resume(request_id, tokens received so far) for up to reconnect_s
    seconds, so the caller sees one uninterrupted, duplicate-free token sequence. Reconnecting
    needs a request id: pass one, or the id echoed in the first chunk is used."""
    req = pb.GenerateRequest(request_id=request_id, prompt=prompt, prompt_token_ids=list(prompt_ids or []),
                             max_new_tokens=max_new_tokens, timeout_s=timeout_s)
    res = GenerateResult(request_id=request_id, status="RUNNING", t_start=time.monotonic())
    call = stub.Generate(req, timeout=rpc_timeout)
    retry_until: float | None = None          # set while re-attaching after a broken stream
    while True:
        outcome, reason = None, ""            # "done" | "broken" | "resubmit"
        try:
            async for chunk in call:
                if chunk.request_id and not res.request_id:
                    res.request_id = chunk.request_id
                if chunk.final:
                    if chunk.status == pb.FAILED and chunk.detail.startswith(SUPERSEDED):
                        outcome, reason = "broken", chunk.detail   # old coordinator was replaced
                    elif chunk.status == pb.NOT_FOUND and not res.tokens and retry_until is not None:
                        outcome = "resubmit"   # the coordinator died before it logged the request
                    else:
                        outcome = "done"
                        res.status = pb.RequestStatus.Name(chunk.status)
                        res.detail = chunk.detail
                        res.server_ttft_ms = chunk.ttft_ms
                        res.server_total_ms = chunk.total_ms
                        res.final_token_ids = list(chunk.token_ids)
                    break
                if retry_until is not None:   # first token after a break: re-attached
                    retry_until = None
                    res.resumed_at.append(time.monotonic())
                res.tokens.append(chunk.token_id)
                res.token_times.append(time.monotonic())
                res.text += chunk.text
                if on_token is not None:
                    on_token(chunk)
                if cancel_after is not None and len(res.tokens) >= cancel_after:
                    call.cancel()
                    outcome, res.status = "done", "CANCELLED"
                    break
            else:
                outcome, reason = "broken", "stream ended without a final message"
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                outcome, res.status = "done", "CANCELLED"
            else:
                outcome, reason = "broken", f"{e.code().name}: {e.details()}"
        except asyncio.CancelledError:
            call.cancel()
            raise
        if outcome == "done":
            if retry_until is not None:
                res.resumed_at.append(time.monotonic())
            break
        if outcome == "resubmit":
            call = stub.Generate(req, timeout=rpc_timeout)
            continue
        now = time.monotonic()
        if retry_until is None:
            if reconnect_s <= 0 or not res.request_id:
                res.status, res.detail = "RPC_ERROR", reason
                break
            retry_until = now + reconnect_s
            res.broken_at.append(now)
            res.reconnects += 1
        elif now > retry_until:
            res.status, res.detail = "RPC_ERROR", f"could not reconnect within {reconnect_s:.0f}s: {reason}"
            break
        await asyncio.sleep(0.05)
        call = stub.Resume(pb.ResumeRequest(request_id=res.request_id, from_step=len(res.tokens)),
                           timeout=rpc_timeout)
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
                             timeout_s=a.timeout, cancel_after=a.cancel_after, reconnect_s=a.reconnect,
                             on_token=lambda c: print(c.text, end="", flush=True))
        print()
        print(f"[status={res.status} tokens={len(res.tokens)} reconnects={res.reconnects} ttft={res.ttft_ms:.1f}ms "
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
    ap.add_argument("--reconnect", type=float, default=0.0,
                    help="if the stream breaks, re-attach with Resume for up to this many seconds")
    ap.add_argument("--stats", action="store_true", help="print coordinator and worker stats")
    ap.add_argument("--reset-stats", action="store_true")
    sys.exit(asyncio.run(_main(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
