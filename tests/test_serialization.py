"""O3: tensor serialisation round-trip is lossless."""

import math

import pytest
import torch

from pllm.proto import pipeline_pb2 as pb
from pllm.serialization import proto_to_tensor, tensor_to_proto


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64,
                                   torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool])
@pytest.mark.parametrize("shape", [(0,), (1,), (7, 768), (2, 3, 4)])
def test_roundtrip_bitwise(dtype, shape):
    g = torch.Generator().manual_seed(0)
    if dtype.is_floating_point:
        t = (torch.randn(shape, generator=g, dtype=torch.float64) * 100).to(dtype)
    elif dtype == torch.bool:
        t = torch.randint(0, 2, shape, generator=g).bool()
    else:
        t = torch.randint(0, 100, shape, generator=g).to(dtype)
    msg = tensor_to_proto(t)
    # go through the actual wire encoding too
    wire = pb.Tensor.FromString(msg.SerializeToString())
    back = proto_to_tensor(wire)
    assert back.dtype == t.dtype and back.shape == t.shape
    assert torch.equal(back.view(torch.uint8) if t.numel() else back, t.view(torch.uint8) if t.numel() else t)


def test_special_values_and_non_contiguous():
    t = torch.tensor([[0.0, -0.0, math.inf], [-math.inf, math.nan, 1e-45]], dtype=torch.float32).t()
    assert not t.is_contiguous()
    back = proto_to_tensor(pb.Tensor.FromString(tensor_to_proto(t).SerializeToString()))
    assert torch.equal(back.view(torch.int32), t.contiguous().view(torch.int32))  # bit patterns incl. NaN, -0.0


def test_activation_bytes_match_shape():
    t = torch.randn(5, 768)
    msg = tensor_to_proto(t)
    assert len(msg.data) == 5 * 768 * 4
    assert len(tensor_to_proto(t, torch.float16).data) == 5 * 768 * 2


def test_fp16_wire_cast_is_lossy_but_close():
    t = torch.randn(4, 768)
    back = proto_to_tensor(tensor_to_proto(t, torch.float16), out_dtype=torch.float32)
    assert back.dtype == torch.float32
    assert torch.allclose(back, t, atol=1e-2, rtol=1e-3)
