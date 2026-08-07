# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu

import triton
import triton.language as tl


pytestmark = pytest.mark.forked
_ASSERT_MESSAGE = "pointer bitcast offset is not divisible by 2"


@triton.jit(do_not_specialize=["byte_offset"])
def pointer_bitcast_scalar_offset_kernel(src, byte_offset, output):
    byte_ptr = src + byte_offset
    bf16_ptr = byte_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    value = tl.load(bf16_ptr)
    tl.store(output, value)


@triton.jit
def pointer_bitcast_tensor_offset_kernel(src, byte_offsets, output, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    offsets = tl.load(byte_offsets + lanes)
    byte_ptrs = src + offsets
    bf16_ptrs = byte_ptrs.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    values = tl.load(bf16_ptrs)
    tl.store(output + lanes, values)


def make_bf16_bytes(values):
    return torch.tensor(values, dtype=torch.bfloat16).view(torch.uint8).npu()


def test_pointer_bitcast_dynamic_scalar_even_offset():
    src = make_bf16_bytes([1.5, -2.25, 3.0])
    output = torch.empty(1, dtype=torch.bfloat16, device="npu")

    pointer_bitcast_scalar_offset_kernel[(1, )](src, 2, output, debug=False)
    torch_npu.npu.synchronize()

    expected = torch.tensor([-2.25], dtype=torch.bfloat16)
    torch.testing.assert_close(output.cpu(), expected)


def test_pointer_bitcast_dynamic_scalar_odd_offset_asserts():
    src = make_bf16_bytes([1.5, -2.25])
    output = torch.empty(1, dtype=torch.bfloat16, device="npu")

    with pytest.raises(RuntimeError, match=_ASSERT_MESSAGE):
        pointer_bitcast_scalar_offset_kernel[(1, )](src, 1, output, debug=False)
        torch_npu.npu.synchronize()


def test_pointer_bitcast_dynamic_tensor_all_offsets_divisible():
    values = [1.5, -2.25, 3.0, 4.5]
    src = make_bf16_bytes(values)
    byte_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device="npu")
    output = torch.empty(4, dtype=torch.bfloat16, device="npu")

    pointer_bitcast_tensor_offset_kernel[(1, )](src, byte_offsets, output, BLOCK=4, debug=False)
    torch_npu.npu.synchronize()

    expected = torch.tensor(values, dtype=torch.bfloat16)
    torch.testing.assert_close(output.cpu(), expected)


def test_pointer_bitcast_dynamic_tensor_one_offset_indivisible_asserts():
    src = make_bf16_bytes([1.5, -2.25, 3.0, 4.5])
    byte_offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device="npu")
    output = torch.empty(4, dtype=torch.bfloat16, device="npu")

    with pytest.raises(RuntimeError, match=_ASSERT_MESSAGE):
        pointer_bitcast_tensor_offset_kernel[(1, )](src, byte_offsets, output, BLOCK=4, debug=False)
        torch_npu.npu.synchronize()
