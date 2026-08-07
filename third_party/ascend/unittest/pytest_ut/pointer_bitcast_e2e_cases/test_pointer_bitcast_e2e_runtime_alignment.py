import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
os.environ.setdefault("ASCEND_VISIBLE_DEVICES", "0")
os.environ.setdefault("NPU_VISIBLE_DEVICES", "0")

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


def _require_npu0():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")
    torch.npu.set_device(0)


def _sync_npu():
    torch.npu.synchronize()


def _write_u32_le(buf, byte_offsets, values):
    for byte_offset, value in zip(byte_offsets, values):
        payload = struct.pack("<I", int(value))
        for byte_idx, byte_value in enumerate(payload):
            buf[int(byte_offset) + byte_idx] = byte_value


def _float_to_bf16_bits(value):
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f32_bits >> 16


def _write_bf16_le(buf, byte_offset, values):
    for idx, value in enumerate(values):
        bits = _float_to_bf16_bits(value)
        buf[int(byte_offset) + idx * 2] = bits & 0xFF
        buf[int(byte_offset) + idx * 2 + 1] = (bits >> 8) & 0xFF


def _ttadapter(compiled):
    assert "ttadapter" in compiled.asm, compiled.asm.keys()
    return compiled.asm["ttadapter"]


def _require_runtime_check(compiled):
    ttadapter = _ttadapter(compiled)
    assert "arith.remsi" in ttadapter, ttadapter
    assert "triton_assert" in ttadapter, ttadapter
    assert "arith.divsi" in ttadapter, ttadapter


def _require_no_runtime_check(compiled):
    ttadapter = _ttadapter(compiled)
    assert "arith.remsi" not in ttadapter, ttadapter
    assert "triton_assert" not in ttadapter, ttadapter


def _require_no_pointer_bitcast_assert(compiled):
    ttadapter = _ttadapter(compiled)
    assert "triton_assert" not in ttadapter, ttadapter


@triton.jit
def _runtime_paged_scale_load(kv_ptr, block_tables_ptr, out_ptr,
                              n_positions, stride_kvblk, stride_kvpos,
                              stride_kvbyte, DIM: tl.constexpr,
                              CACHE_BLOCK_SIZE: tl.constexpr,
                              BLOCK: tl.constexpr):
    kv_global_pos = tl.arange(0, BLOCK)
    valid = kv_global_pos < n_positions
    logical_block = kv_global_pos // CACHE_BLOCK_SIZE
    intra_block_pos = kv_global_pos % CACHE_BLOCK_SIZE
    physical_block = tl.load(block_tables_ptr + logical_block,
                             mask=valid,
                             other=0)
    kv_base = (physical_block * stride_kvblk
               + intra_block_pos * stride_kvpos)
    scale_addr = kv_base + DIM * stride_kvbyte
    scale_ptr = (kv_ptr + scale_addr).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(scale_ptr, mask=valid, other=0)
    tl.store(out_ptr + kv_global_pos, scale_u32, mask=valid)


@triton.jit
def _runtime_scalar_bf16_load(src, byte_offset_ptr, out,
                              N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    byte_offset = tl.load(byte_offset_ptr)
    bf16_ptr = (src + byte_offset).to(
        tl.pointer_type(tl.bfloat16), bitcast=True)
    values = tl.load(bf16_ptr + lanes, mask=lanes < N, other=0.0)
    tl.store(out + lanes, values, mask=lanes < N)


@triton.jit
def _runtime_tensor_u32_load(src, byte_offsets_ptr, out,
                             N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    u32_ptrs = (src + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_tensor_u32_store(src, dst, byte_offsets_ptr,
                              N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    values = tl.load(src + lanes, mask=mask, other=0)
    u32_ptrs = (dst + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(u32_ptrs, values, mask=mask)


@triton.jit
def _runtime_2d_tensor_u32_load(src, byte_offsets_ptr, out,
                                BLOCK_M: tl.constexpr,
                                BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    linear = rows * BLOCK_N + cols
    byte_offsets = tl.load(byte_offsets_ptr + linear)
    u32_ptrs = (src + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs)
    tl.store(out + linear, values)


@triton.jit
def _runtime_2d_tensor_u32_store(src, dst, byte_offsets_ptr,
                                 BLOCK_M: tl.constexpr,
                                 BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    linear = rows * BLOCK_N + cols
    byte_offsets = tl.load(byte_offsets_ptr + linear)
    values = tl.load(src + linear)
    u32_ptrs = (dst + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    tl.store(u32_ptrs, values)


@triton.jit
def _runtime_pointer_then_value_bitcast(src, byte_offsets_ptr, out,
                                        N: tl.constexpr,
                                        BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = tl.load(byte_offsets_ptr + lanes, mask=mask, other=0)
    u32_ptrs = (src + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(u32_ptrs, mask=mask, other=0)
    scale_f32 = scale_u32.to(tl.float32, bitcast=True)
    tl.store(out + lanes, scale_f32, mask=mask)


@triton.jit
def _loaded_offset_division_without_pointer_bitcast(src, offsets_ptr, out,
                                                    N: tl.constexpr,
                                                    DIVISOR: tl.constexpr,
                                                    BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    loaded_offsets = tl.load(offsets_ptr + lanes, mask=mask, other=0)
    element_offsets = loaded_offsets // DIVISOR
    values = tl.load(src + element_offsets, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_scalar_bf16_store(src, dst, byte_offset_ptr,
                               N: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offset = tl.load(byte_offset_ptr)
    values = tl.load(src + lanes, mask=mask, other=0.0)
    bf16_ptr = (dst + byte_offset).to(
        tl.pointer_type(tl.bfloat16), bitcast=True)
    tl.store(bf16_ptr + lanes, values, mask=mask)


@triton.jit
def _static_axisinfo_u32_load(src, out, N: tl.constexpr,
                              BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_offsets = lanes * 4
    u32_ptrs = (src + byte_offsets).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    values = tl.load(u32_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_multi_addptr_bf16_load(src, params_ptr, out,
                                    N: tl.constexpr, BLOCK: tl.constexpr):
    block_idx = tl.load(params_ptr + 0)
    token_pos = tl.load(params_ptr + 1)
    block_stride = tl.load(params_ptr + 2)
    token_bytes = tl.load(params_ptr + 3)
    bf16_offset = tl.load(params_ptr + 4)
    block_base = src + block_idx * block_stride
    token_base = block_base + token_pos * token_bytes
    bf16_ptr = (token_base + bf16_offset).to(
        tl.pointer_type(tl.bfloat16), bitcast=True)
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    values = tl.load(bf16_ptr + lanes, mask=mask, other=0.0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _runtime_scalar_u32_load(src, byte_offset_ptr, out):
    byte_offset = tl.load(byte_offset_ptr)
    u32_ptr = (src + byte_offset).to(
        tl.pointer_type(tl.uint32, 1), bitcast=True)
    value = tl.load(u32_ptr)
    tl.store(out, value)


def test_runtime_aligned_paged_mqa_scale_address():
    _require_npu0()
    cache_block_size = 4
    dim = 8
    record_bytes = dim + 4
    stride_kvpos = record_bytes
    stride_kvblk = cache_block_size * record_bytes
    stride_kvbyte = 1
    n_positions = 8
    block = 8
    block_table = [2, 0]
    values = [100 + idx for idx in range(n_positions)]

    host = torch.zeros(3 * stride_kvblk, dtype=torch.uint8)
    byte_offsets = []
    for logical_pos in range(n_positions):
        logical_block = logical_pos // cache_block_size
        intra_block_pos = logical_pos % cache_block_size
        physical_block = block_table[logical_block]
        byte_offsets.append(physical_block * stride_kvblk
                            + intra_block_pos * stride_kvpos + dim)
    _write_u32_le(host, byte_offsets, values)

    kv = host.npu()
    block_tables = torch.tensor(block_table, dtype=torch.int32).npu()
    out = torch.empty((block,), device="npu", dtype=torch.int32)
    args = (kv, block_tables, out, n_positions, stride_kvblk,
            stride_kvpos, stride_kvbyte)
    meta = {"DIM": dim, "CACHE_BLOCK_SIZE": cache_block_size,
            "BLOCK": block}

    compiled = _runtime_paged_scale_load.warmup(
        *args, **meta, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_paged_scale_load[(1,)](*args, **meta)
    _sync_npu()

    torch.testing.assert_close(
        out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_aligned_scalar_bf16_load():
    _require_npu0()
    byte_offset = 18
    values = [1.0, 2.0, 4.0, 8.0]
    host = torch.zeros(64, dtype=torch.uint8)
    _write_bf16_le(host, byte_offset, values)
    src = host.npu()
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()
    out = torch.empty((len(values),), device="npu", dtype=torch.bfloat16)

    compiled = _runtime_scalar_bf16_load.warmup(
        src, offset, out, N=len(values), BLOCK=8, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_scalar_bf16_load[(1,)](
        src, offset, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_aligned_tensor_offsets_u32_load():
    _require_npu0()
    byte_offsets = [4, 20, 36, 52]
    values = [0x01020304, 0x11121314, 0x21222324, 0x31323334]
    host = torch.zeros(80, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, values)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(values),), device="npu", dtype=torch.int32)

    compiled = _runtime_tensor_u32_load.warmup(
        src, offsets, out, N=len(values), BLOCK=4, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_tensor_u32_load[(1,)](
        src, offsets, out, N=len(values), BLOCK=4)
    _sync_npu()

    torch.testing.assert_close(
        out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_aligned_tensor_offsets_u32_store():
    _require_npu0()
    byte_offsets = [8, 24, 40, 56]
    values = [101, 202, 303, 404]
    src = torch.tensor(values, dtype=torch.int32).npu()
    dst = torch.zeros(80, device="npu", dtype=torch.uint8)
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()

    compiled = _runtime_tensor_u32_store.warmup(
        src, dst, offsets, N=len(values), BLOCK=4, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_tensor_u32_store[(1,)](
        src, dst, offsets, N=len(values), BLOCK=4)
    _sync_npu()

    expected = torch.zeros(80, dtype=torch.uint8)
    _write_u32_le(expected, byte_offsets, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_runtime_aligned_2d_tensor_offsets_u32_load():
    _require_npu0()
    block_m = 2
    block_n = 4
    byte_offsets = [0, 12, 24, 36, 48, 60, 72, 84]
    values = [11, 22, 33, 44, 55, 66, 77, 88]
    host = torch.zeros(96, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, values)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((block_m, block_n), device="npu", dtype=torch.int32)

    compiled = _runtime_2d_tensor_u32_load.warmup(
        src, offsets, out, BLOCK_M=block_m, BLOCK_N=block_n, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_2d_tensor_u32_load[(1,)](
        src, offsets, out, BLOCK_M=block_m, BLOCK_N=block_n)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.int32).reshape(
        block_m, block_n)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_aligned_2d_tensor_offsets_u32_store():
    _require_npu0()
    block_m = 2
    block_n = 4
    byte_offsets = [4, 16, 28, 40, 52, 64, 76, 88]
    values = [111, 222, 333, 444, 555, 666, 777, 888]
    src = torch.tensor(values, dtype=torch.int32).reshape(
        block_m, block_n).npu()
    dst = torch.zeros(96, device="npu", dtype=torch.uint8)
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()

    compiled = _runtime_2d_tensor_u32_store.warmup(
        src, dst, offsets, BLOCK_M=block_m, BLOCK_N=block_n, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_2d_tensor_u32_store[(1,)](
        src, dst, offsets, BLOCK_M=block_m, BLOCK_N=block_n)
    _sync_npu()

    expected = torch.zeros(96, dtype=torch.uint8)
    _write_u32_le(expected, byte_offsets, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_runtime_pointer_bitcast_followed_by_value_bitcast():
    _require_npu0()
    byte_offsets = [4, 12, 20, 28]
    values = [1.0, -2.0, 0.5, 8.0]
    bit_patterns = [
        struct.unpack("<I", struct.pack("<f", value))[0] for value in values
    ]
    host = torch.zeros(40, dtype=torch.uint8)
    _write_u32_le(host, byte_offsets, bit_patterns)
    src = host.npu()
    offsets = torch.tensor(byte_offsets, dtype=torch.int64).npu()
    out = torch.empty((len(values),), device="npu", dtype=torch.float32)

    compiled = _runtime_pointer_then_value_bitcast.warmup(
        src, offsets, out, N=len(values), BLOCK=4, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_pointer_then_value_bitcast[(1,)](
        src, offsets, out, N=len(values), BLOCK=4)
    _sync_npu()

    torch.testing.assert_close(
        out.cpu(), torch.tensor(values, dtype=torch.float32), rtol=0, atol=0)


def test_loaded_tensor_offset_division_without_pointer_bitcast_is_unchanged():
    _require_npu0()
    loaded_offsets = [0, 2, 6, 10]
    source = torch.arange(16, dtype=torch.int32).npu()
    offsets = torch.tensor(loaded_offsets, dtype=torch.int64).npu()
    out = torch.empty(
        (len(loaded_offsets),), device="npu", dtype=torch.int32)

    compiled = _loaded_offset_division_without_pointer_bitcast.warmup(
        source, offsets, out, N=len(loaded_offsets), DIVISOR=2, BLOCK=4,
        grid=(1,))
    _require_no_pointer_bitcast_assert(compiled)
    _loaded_offset_division_without_pointer_bitcast[(1,)](
        source, offsets, out, N=len(loaded_offsets), DIVISOR=2, BLOCK=4)
    _sync_npu()

    expected = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_aligned_scalar_bf16_store():
    _require_npu0()
    byte_offset = 22
    values = [3.0, 6.0, 9.0, 12.0]
    src = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).npu()
    dst = torch.zeros(64, device="npu", dtype=torch.uint8)
    offset = torch.tensor([byte_offset], dtype=torch.int64).npu()

    compiled = _runtime_scalar_bf16_store.warmup(
        src, dst, offset, N=len(values), BLOCK=8, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_scalar_bf16_store[(1,)](
        src, dst, offset, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.zeros(64, dtype=torch.uint8)
    _write_bf16_le(expected, byte_offset, values)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_static_axisinfo_proven_ratio4_has_no_runtime_assert():
    _require_npu0()
    block = 4
    values = [11, 22, 33, 44]
    host = torch.zeros(block * 4, dtype=torch.uint8)
    _write_u32_le(host, [idx * 4 for idx in range(block)], values)
    src = host.npu()
    out = torch.empty((block,), device="npu", dtype=torch.int32)

    compiled = _static_axisinfo_u32_load.warmup(
        src, out, N=block, BLOCK=block, grid=(1,))
    _require_no_runtime_check(compiled)
    _static_axisinfo_u32_load[(1,)](
        src, out, N=block, BLOCK=block)
    _sync_npu()

    torch.testing.assert_close(
        out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_runtime_aligned_multi_addptr_bf16_load():
    _require_npu0()
    params_host = [1, 1, 64, 16, 8]
    byte_offset = 64 + 16 + 8
    values = [2.0, 4.0, 6.0, 8.0]
    host = torch.zeros(128, dtype=torch.uint8)
    _write_bf16_le(host, byte_offset, values)
    src = host.npu()
    params = torch.tensor(params_host, dtype=torch.int64).npu()
    out = torch.empty((len(values),), device="npu", dtype=torch.bfloat16)

    compiled = _runtime_multi_addptr_bf16_load.warmup(
        src, params, out, N=len(values), BLOCK=8, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_multi_addptr_bf16_load[(1,)](
        src, params, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_runtime_aligned_negative_scalar_offset_u32_load():
    _require_npu0()
    host = torch.zeros(64, dtype=torch.uint8)
    value = 0x10203040
    _write_u32_le(host, [12], [value])
    storage = host.npu()
    src = storage[16:]
    offset = torch.tensor([-4], dtype=torch.int64).npu()
    out = torch.empty((), device="npu", dtype=torch.int32)

    compiled = _runtime_scalar_u32_load.warmup(
        src, offset, out, grid=(1,))
    _require_runtime_check(compiled)
    _runtime_scalar_u32_load[(1,)](src, offset, out)
    _sync_npu()

    assert int(out.cpu()) == value


def _run_runtime_assert_worker(case_name):
    worker = Path(__file__).with_name("pointer_bitcast_runtime_worker.py")
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    env["ASCEND_VISIBLE_DEVICES"] = "0"
    env["NPU_VISIBLE_DEVICES"] = "0"
    env["TRITON_DEBUG"] = "0"
    completed = subprocess.run(
        [sys.executable, str(worker), case_name],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr

    assert "COMPILE_OK" in output, output
    assert "COMPILE_REJECTED" not in output, output
    assert "ASSERT_NOT_TRIGGERED" not in output, output
    hard_runtime_failure = completed.returncode not in {0, 2, 3}
    runtime_assert_observed = (
        "RUNTIME_ASSERT_OBSERVED" in output or hard_runtime_failure)
    assert runtime_assert_observed, output
    assert re.search(
        r"pointer bitcast offset|not divisible by [24]|divisible by [24]|alignment",
        output,
        flags=re.IGNORECASE,
    ), output


def test_runtime_misaligned_scalar_offset_reports_device_assert():
    _require_npu0()
    _run_runtime_assert_worker("scalar")


def test_runtime_misaligned_tensor_lane_reports_device_assert():
    _require_npu0()
    _run_runtime_assert_worker("tensor")


def test_runtime_misaligned_multi_addptr_reports_device_assert():
    _require_npu0()
    _run_runtime_assert_worker("multi_addptr")


def test_runtime_misaligned_masked_off_lane_reports_device_assert():
    _require_npu0()
    _run_runtime_assert_worker("masked_tensor")


def test_runtime_misaligned_negative_offset_reports_device_assert():
    _require_npu0()
    _run_runtime_assert_worker("negative_scalar")
