"""Lossless conversion between torch tensors and the `Tensor` protobuf message."""

from __future__ import annotations

import sys

import torch

from pllm.proto import pipeline_pb2 as pb

assert sys.byteorder == "little", "wire format is little-endian"

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_NAMES = {v: k for k, v in _DTYPES.items()}


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}") from None


def tensor_to_proto(t: torch.Tensor, wire_dtype: torch.dtype | None = None) -> pb.Tensor:
    """Serialise a tensor. If `wire_dtype` is given the tensor is cast first (lossy for fp16)."""
    if wire_dtype is not None and t.dtype != wire_dtype:
        t = t.to(wire_dtype)
    t = t.detach().contiguous().cpu()
    # Reinterpret as raw bytes; works for every dtype including bfloat16 (no numpy equivalent).
    data = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
    return pb.Tensor(shape=list(t.shape), dtype=_NAMES[t.dtype], data=data)


def proto_to_tensor(msg: pb.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    dtype = dtype_from_name(msg.dtype)
    shape = tuple(msg.shape)
    if len(msg.data) == 0:
        t = torch.empty(shape, dtype=dtype)
    else:
        # bytearray gives torch.frombuffer a writable buffer it can own.
        t = torch.frombuffer(bytearray(msg.data), dtype=dtype).reshape(shape)
    if out_dtype is not None and t.dtype != out_dtype:
        t = t.to(out_dtype)
    return t
