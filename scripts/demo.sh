#!/usr/bin/env bash
# O9 demo: normal generation, overload, and a worker kill with recovery - in one terminal.
# Record it with e.g.  asciinema rec demo.cast -c "bash scripts/demo.sh"   (or any screen recorder).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-50050}"
ADDR="127.0.0.1:${PORT}"
mkdir -p logs

say() { printf '\n\033[1;34m== %s ==\033[0m\n' "$*"; }

say "Starting coordinator (spawns worker 1 = stage 0 and worker 2 = stage 1)"
"$PY" -m pllm.coordinator --port "$PORT" --admission-queue 8 --max-active 8 > logs/demo_coordinator.log 2>&1 &
COORD=$!
trap 'kill $COORD 2>/dev/null; wait $COORD 2>/dev/null' EXIT
until grep -q "COORDINATOR READY" logs/demo_coordinator.log; do sleep 0.5; done
"$PY" -m pllm.client --addr "$ADDR" --stats | "$PY" -c "import json,sys; s=json.load(sys.stdin); print('epochs', s['epochs'], 'worker pids', s['worker_pids'])"

say "1) Normal generation (tokens stream as they are produced)"
"$PY" -m pllm.client --addr "$ADDR" --prompt "The capital of France is" --max-new-tokens 24

say "2) Overload: 60 simultaneous requests against an admission queue of 8"
"$PY" - "$ADDR" <<'PYEOF'
import asyncio, sys, collections
from pllm.client import channel, generate
from pllm.proto import pipeline_pb2_grpc as g
async def main(addr):
    async with channel(addr) as ch:
        stub = g.CoordinatorServiceStub(ch)
        rs = await asyncio.gather(*[generate(stub, prompt="Once upon a time", max_new_tokens=16) for _ in range(60)])
    c = collections.Counter(r.status for r in rs)
    print("outcomes:", dict(c))
    rej = [r for r in rs if r.status == "OVERLOADED"]
    if rej:
        print(f"example rejection after {rej[0].latency_ms:.1f} ms: {rej[0].detail}")
asyncio.run(main(sys.argv[1]))
PYEOF

say "3) Kill worker 2 (stage 1) with SIGKILL while a request is streaming"
PID=$("$PY" -m pllm.client --addr "$ADDR" --stats | "$PY" -c "import json,sys; print(json.load(sys.stdin)['worker_pids']['1'])")
( sleep 1.0; echo; echo "   >>> kill -9 $PID (worker 2)"; kill -9 "$PID" ) &
"$PY" -m pllm.client --addr "$ADDR" --max-new-tokens 120 \
  --prompt "It was a dark and stormy night; the rain fell in torrents, and"
"$PY" -m pllm.client --addr "$ADDR" --stats | "$PY" -c "
import json,sys; s=json.load(sys.stdin); ev=s['recovery_events'][-1]
print('epochs now', s['epochs'])
print(f\"recovery: heartbeat gap {ev['detection_ms']:.0f} ms, restart {ev['restart_ms']:.0f} ms, replay {ev['replay_ms']:.0f} ms\")"

say "Done"
