"""Per-request KV-cache management for one pipeline stage.

Each worker keeps, for every in-flight request, the attention keys/values of
*its own* layers only. The cache is soft state: it can always be rebuilt by
replaying the prompt plus the tokens generated so far (see coordinator recovery).

Besides the tensors, each entry remembers the idempotency key of the last step
it applied, (attempt, step), together with that step's output. This lets the
stage enforce strict in-order application and answer a duplicated step from the
cache instead of appending its keys/values twice.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

KV = tuple[torch.Tensor, torch.Tensor]  # (key, value), each [1, n_head, seq_len, head_dim]


class CacheFullError(Exception):
    pass


@dataclass
class RequestCache:
    request_id: str
    attempt: int
    layers: list[KV | None]
    length: int = 0                 # number of tokens whose K/V are stored
    last_step: int = -1             # last step applied (-1 = none yet)
    last_output: object = None      # output of last_step (hidden tensor or token id), for idempotent replays
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)

    def nbytes(self) -> int:
        total = 0
        for kv in self.layers:
            if kv is not None:
                total += kv[0].numel() * kv[0].element_size() + kv[1].numel() * kv[1].element_size()
        return total


@dataclass
class Tombstone:
    """Remembers a request that finished on this worker, so late messages for it are rejected
    (or, for a retry of the final step, answered idempotently)."""
    attempt: int
    last_step: int
    last_output: object
    reason: str  # "completed" | "released" | "evicted"


class KVCacheManager:
    def __init__(self, num_layers: int, max_requests: int = 128, ttl_s: float = 120.0,
                 max_tombstones: int = 4096):
        self.num_layers = num_layers
        self.max_requests = max_requests
        self.ttl_s = ttl_s
        self.max_tombstones = max_tombstones
        self._caches: dict[str, RequestCache] = {}
        self._tombstones: OrderedDict[str, Tombstone] = OrderedDict()
        self.evictions = 0

    # -- lookup -------------------------------------------------------------
    def get(self, request_id: str) -> RequestCache | None:
        c = self._caches.get(request_id)
        if c is not None:
            c.last_access = time.monotonic()
        return c

    def tombstone(self, request_id: str) -> Tombstone | None:
        return self._tombstones.get(request_id)

    def __len__(self) -> int:
        return len(self._caches)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._caches

    def request_ids(self) -> list[str]:
        return list(self._caches)

    def total_bytes(self) -> int:
        return sum(c.nbytes() for c in self._caches.values())

    # -- mutation -----------------------------------------------------------
    def create(self, request_id: str, attempt: int) -> RequestCache:
        """Create (or replace) the cache for a request. Raises CacheFullError at capacity."""
        self._caches.pop(request_id, None)
        self._tombstones.pop(request_id, None)
        if len(self._caches) >= self.max_requests:
            self.evict_expired()
        if len(self._caches) >= self.max_requests:
            raise CacheFullError(f"KV cache full ({self.max_requests} requests)")
        c = RequestCache(request_id=request_id, attempt=attempt, layers=[None] * self.num_layers)
        self._caches[request_id] = c
        return c

    def release(self, request_id: str, reason: str = "released") -> bool:
        c = self._caches.pop(request_id, None)
        if c is None:
            if request_id not in self._tombstones:
                self._add_tombstone(request_id, Tombstone(-1, -1, None, reason))
            return False
        self._add_tombstone(request_id, Tombstone(c.attempt, c.last_step, c.last_output, reason))
        return True

    def evict_expired(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        expired = [rid for rid, c in self._caches.items() if now - c.last_access > self.ttl_s]
        for rid in expired:
            self.release(rid, reason="evicted")
        self.evictions += len(expired)
        return len(expired)

    def clear(self) -> None:
        self._caches.clear()
        self._tombstones.clear()

    def _add_tombstone(self, request_id: str, t: Tombstone) -> None:
        self._tombstones[request_id] = t
        self._tombstones.move_to_end(request_id)
        while len(self._tombstones) > self.max_tombstones:
            self._tombstones.popitem(last=False)

    # -- tensor helpers -----------------------------------------------------
    @staticmethod
    def append(cache: RequestCache, layer: int, k: torch.Tensor, v: torch.Tensor) -> KV:
        """Append new keys/values for one layer and return the full (past + new) K/V."""
        past = cache.layers[layer]
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        cache.layers[layer] = (k, v)
        return k, v
