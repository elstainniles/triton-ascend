import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
os.environ.setdefault("ASCEND_VISIBLE_DEVICES", "0")
os.environ.setdefault("NPU_VISIBLE_DEVICES", "0")

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _runtime_scalar_u32(src, byte_offset_ptr, out):
    byte_offset = tl.load(byte_offset_ptr)
    ptr = (src + byte_offset).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    value = tl.load(ptr)
    tl.store(out, value)


@triton.jit
def _runtime_tensor_u32(src, byte_offsets_ptr, out,
                        ACTIVE_N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    byte_offsets = tl.load(byte_offsets_ptr + lanes)
    ptrs = (src + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    mask = lanes < ACTIVE_N
    values = tl.load(ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_multi_addptr_bf16(src, params_ptr, out):
    block_idx = tl.load(params_ptr + 0)
    token_pos = tl.load(params_ptr + 1)
    block_stride = tl.load(params_ptr + 2)
    token_bytes = tl.load(params_ptr + 3)
    bf16_offset = tl.load(params_ptr + 4)
    block_base = src + block_idx * block_stride
    token_base = block_base + token_pos * token_bytes
    ptr = (token_base + bf16_offset).to(
        tl.pointer_type(tl.bfloat16), bitcast=True)
    value = tl.load(ptr)
    tl.store(out, value)


def _prepare_scalar():
    src = torch.zeros((64,), device="npu", dtype=torch.uint8)
    offset = torch.tensor([2], device="npu", dtype=torch.int64)
    out = torch.empty((), device="npu", dtype=torch.int32)
    return _runtime_scalar_u32, (src, offset, out), {}


def _prepare_tensor(active_n):
    src = torch.zeros((64,), device="npu", dtype=torch.uint8)
    offsets = torch.tensor(
        [0, 4, 6, 12], device="npu", dtype=torch.int64)
    out = torch.empty((4,), device="npu", dtype=torch.int32)
    return (_runtime_tensor_u32, (src, offsets, out),
            {"ACTIVE_N": active_n, "BLOCK": 4})


def _prepare_multi_addptr():
    src = torch.zeros((128,), device="npu", dtype=torch.uint8)
    params = torch.tensor(
        [1, 1, 64, 16, 7], device="npu", dtype=torch.int64)
    out = torch.empty((), device="npu", dtype=torch.bfloat16)
    return _runtime_multi_addptr_bf16, (src, params, out), {}


def _prepare_negative_scalar():
    storage = torch.zeros((64,), device="npu", dtype=torch.uint8)
    src = storage[16:]
    offset = torch.tensor([-2], device="npu", dtype=torch.int64)
    out = torch.empty((), device="npu", dtype=torch.int32)
    return _runtime_scalar_u32, (src, offset, out), {}


def _prepare(case_name):
    if case_name == "scalar":
        return _prepare_scalar()
    if case_name == "tensor":
        return _prepare_tensor(active_n=4)
    if case_name == "masked_tensor":
        return _prepare_tensor(active_n=2)
    if case_name == "multi_addptr":
        return _prepare_multi_addptr()
    if case_name == "negative_scalar":
        return _prepare_negative_scalar()
    raise ValueError(case_name)


def main():
    valid_cases = {
        "scalar", "tensor", "masked_tensor", "multi_addptr",
        "negative_scalar",
    }
    if len(sys.argv) != 2 or sys.argv[1] not in valid_cases:
        print("usage: pointer_bitcast_runtime_worker.py "
              + "|".join(sorted(valid_cases)))
        return 64
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("NPU_UNAVAILABLE")
        return 77
    torch.npu.set_device(0)
    kernel, args, constexprs = _prepare(sys.argv[1])

    try:
        kernel.warmup(*args, grid=(1,), **constexprs)
    except Exception as exc:
        print("COMPILE_REJECTED", flush=True)
        print(str(exc), flush=True)
        traceback.print_exc()
        return 2

    print("COMPILE_OK", flush=True)
    try:
        kernel[(1,)](*args, **constexprs)
        torch.npu.synchronize()
    except Exception as exc:
        print("RUNTIME_ASSERT_OBSERVED", flush=True)
        print(str(exc), flush=True)
        traceback.print_exc()
        return 0

    print("ASSERT_NOT_TRIGGERED", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
