"""Shared configuration: defaults, paths and the coordinator/worker config objects."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "prompts.json"
REFERENCE_FILE = REPO_ROOT / "reference_outputs.json"
LOG_DIR = REPO_ROOT / "logs"

MODEL_NAME = os.environ.get("PLLM_MODEL", "gpt2")
MODEL_REVISION = os.environ.get("PLLM_MODEL_REVISION", "main")

# Torch intra-op threads per process. The baseline and every worker use the same
# value so that matmul reduction orders (and therefore floating point results) match.
DEFAULT_THREADS = int(os.environ.get("PLLM_THREADS", "4"))

DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_SEED = 0

COORDINATOR_PORT = 50050
WORKER_BASE_PORT = 50051  # stage i listens on WORKER_BASE_PORT + i

# gRPC messages carry activations; raise the default 4 MB limit for long prefills.
GRPC_MAX_MESSAGE_BYTES = 256 * 1024 * 1024
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.so_reuseport", 0),
    # A restarted worker must be reachable again quickly: cap the reconnect backoff.
    ("grpc.initial_reconnect_backoff_ms", 50),
    ("grpc.min_reconnect_backoff_ms", 50),
    ("grpc.max_reconnect_backoff_ms", 250),
]


@dataclass
class WorkerConfig:
    stage: int
    num_stages: int
    port: int
    split: str = "6"                 # layer boundaries, e.g. "6" -> [0,6) [6,12); "auto" = memory balanced
    next_addr: str | None = None     # address of stage+1 (None for the last stage)
    epoch: int = 1
    host: str = "127.0.0.1"
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    threads: int = DEFAULT_THREADS
    queue_size: int = 8              # bounded input queue (Forward calls waiting for / using compute)
    max_cached_requests: int = 128   # KV-cache capacity (requests)
    cache_ttl_s: float = 120.0       # idle KV caches are evicted after this long
    downstream_timeout_s: float = 10.0
    activation_dtype: str = "float32"  # wire dtype for activations ("float16" = compression)
    link_delay_ms: float = 0.0       # artificial delay on the link to the next stage (experiment E9)
    parent_pid: int = 0              # exit when this process disappears (set when spawned by the coordinator)


@dataclass
class CoordinatorConfig:
    port: int = COORDINATOR_PORT
    host: str = "127.0.0.1"
    num_stages: int = 2
    split: str = "6"
    worker_addrs: list[str] = field(default_factory=list)  # empty -> spawn local workers
    spawn_workers: bool = True
    worker_base_port: int = WORKER_BASE_PORT
    worker_threads: int = DEFAULT_THREADS
    worker_queue_size: int = 8
    worker_max_cached_requests: int = 128
    activation_dtype: str = "float32"

    # Admission control / backpressure
    admission_queue_size: int = 64   # requests waiting for an execution slot
    max_active: int = 64             # requests holding KV cache / being decoded (<= worker cache cap)

    # Micro-batching / pipelining
    kv_cache: bool = True
    max_batch_size: int = 8
    max_wait_ms: float = 2.0
    max_inflight_batches: int = 2    # >1 enables pipelining across stages

    # Timeouts
    rpc_timeout_s: float = 10.0
    request_timeout_s: float = 300.0

    # Failure detection / recovery
    heartbeat_interval_s: float = 0.1
    heartbeat_rpc_timeout_s: float = 0.25
    failure_timeout_s: float = 0.6   # declare a worker dead after this long without a heartbeat
    recovery_deadline_s: float = 30.0
    max_retries: int = 20

    # Fault injection (experiment E8)
    duplicate_dispatch: bool = False  # re-send every micro-batch a second time

    def to_dict(self) -> dict:
        return asdict(self)


def parse_split(split: str, n_layer: int, num_stages: int, layer_params: list[int] | None = None,
                embed_params: int = 0, head_params: int = 0) -> list[tuple[int, int]]:
    """Turn a split spec into per-stage [start, end) layer ranges.

    "6"      -> [(0, 6), (6, 12)]
    "4,8"    -> [(0, 4), (4, 8), (8, 12)]
    "auto"   -> boundaries chosen to balance parameter memory, accounting for the
                embedding table on the first stage and the LM head on the last stage.
    """
    if split == "auto":
        if num_stages == 1:
            return [(0, n_layer)]
        assert layer_params is not None
        return _balanced_split(n_layer, num_stages, layer_params, embed_params, head_params)
    if split in ("", "none"):
        bounds: list[int] = []
    else:
        bounds = [int(x) for x in split.split(",")]
    if len(bounds) != num_stages - 1:
        # Even split if the spec does not match the number of stages.
        step = n_layer / num_stages
        bounds = [round(step * i) for i in range(1, num_stages)]
    edges = [0] + bounds + [n_layer]
    ranges = [(edges[i], edges[i + 1]) for i in range(num_stages)]
    for start, end in ranges:
        if not 0 <= start < end <= n_layer:
            raise ValueError(f"invalid split {split!r} for {n_layer} layers / {num_stages} stages")
    return ranges


def _balanced_split(n_layer, num_stages, layer_params, embed_params, head_params):
    # Exhaustive search over boundaries is cheap for small layer counts; for N stages
    # use a greedy prefix walk that minimises the maximum stage size.
    import itertools

    best, best_cost = None, None
    for bounds in itertools.combinations(range(1, n_layer), num_stages - 1):
        edges = (0,) + bounds + (n_layer,)
        sizes = []
        for i in range(num_stages):
            s = sum(layer_params[edges[i]:edges[i + 1]])
            if i == 0:
                s += embed_params
            if i == num_stages - 1:
                s += head_params
            sizes.append(s)
        cost = max(sizes)
        if best_cost is None or cost < best_cost:
            best, best_cost = edges, cost
    return [(best[i], best[i + 1]) for i in range(num_stages)]
