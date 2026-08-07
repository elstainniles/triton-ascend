import os
import struct

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
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _pack_i16(value):
    return struct.pack("<h", int(value))


def _pack_i32(value):
    return struct.pack("<i", int(value))


def _pack_f32(value):
    return struct.pack("<f", float(value))


def _float_to_bf16_bits(value):
    f32_bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f32_bits >> 16


def _pack_bf16(value):
    bits = _float_to_bf16_bits(value)
    return bytes((bits & 0xFF, (bits >> 8) & 0xFF))


def _write_packed_values(buf, byte_offsets, values, pack_fn):
    for byte_offset, value in zip(byte_offsets, values):
        payload = pack_fn(value)
        start = int(byte_offset)
        for byte_idx, byte_value in enumerate(payload):
            buf[start + byte_idx] = byte_value


def _expected_bytes(values, pack_fn):
    out = []
    for value in values:
        out.extend(pack_fn(value))
    return torch.tensor(out, dtype=torch.uint8)


def _float_bits_to_i32(value):
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


@triton.jit
def _load_widen_from_u8(src, out, n_elements, BASE: tl.constexpr,
                        BYTE_WIDTH: tl.constexpr, TARGET_DTYPE: tl.constexpr,
                        BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    byte_offsets = BASE + lanes * BYTE_WIDTH
    target_ptrs = (src + byte_offsets).to(tl.pointer_type(TARGET_DTYPE, 1),
                                          bitcast=True)
    values = tl.load(target_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _store_widen_to_u8(src, dst, n_elements, BASE: tl.constexpr,
                       BYTE_WIDTH: tl.constexpr, TARGET_DTYPE: tl.constexpr,
                       BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < n_elements
    values = tl.load(src + lanes, mask=mask, other=0)
    byte_offsets = BASE + lanes * BYTE_WIDTH
    target_ptrs = (dst + byte_offsets).to(tl.pointer_type(TARGET_DTYPE, 1),
                                          bitcast=True)
    tl.store(target_ptrs, values, mask=mask)


@triton.jit
def _same_width_i32_pointer_bitcast(src, out, word_base: tl.constexpr,
                                    n_elements: tl.constexpr,
                                    BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    src_words = src + word_base
    f32_ptr = src_words.to(tl.pointer_type(tl.float32), bitcast=True)
    values = tl.load(f32_ptr + lanes, mask=lanes < n_elements, other=0.0)
    tl.store(out + lanes, values)


@triton.jit
def _value_bitcast_i32_to_f32(src, out, n_elements: tl.constexpr,
                              BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    raw = tl.load(src + lanes, mask=lanes < n_elements, other=0)
    values = raw.to(tl.float32, bitcast=True)
    tl.store(out + lanes, values)


@triton.jit
def _narrow_i32_to_u8_load(src, out, word_base: tl.constexpr,
                           n_bytes: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    src_words = src + word_base
    byte_ptr = src_words.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptr + lanes, mask=lanes < n_bytes, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _narrow_i32_tensor_ptrs_to_u8_load(src, word_offsets_ptr, out,
                                       N: tl.constexpr,
                                       BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    word_offsets = tl.load(word_offsets_ptr + lanes, mask=mask, other=0)
    word_ptrs = src + word_offsets
    byte_ptrs = word_ptrs.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptrs, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _legacy_i1_pointer_to_u8_load(src, out, N: tl.constexpr,
                                  BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < N
    byte_ptr = src.to(tl.pointer_type(tl.uint8), bitcast=True)
    values = tl.load(byte_ptr + lanes, mask=mask, other=0)
    tl.store(out + lanes, values, mask=mask)


@triton.jit
def _plain_u8_copy(src, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    values = tl.load(src + lanes, mask=lanes < n_elements, other=0)
    tl.store(out + lanes, values)


@triton.jit
def _misaligned_u8_to_i32(src, out):
    ptr = (src + 2).to(tl.pointer_type(tl.int32, 1), bitcast=True)
    value = tl.load(ptr)
    tl.store(out, value)


@pytest.mark.parametrize(
    "name,byte_width,target_dtype,torch_dtype,values,pack_fn",
    [
        ("i16", 2, tl.int16, torch.int16, [11, 22, 33, 44, 55], _pack_i16),
        ("i32", 4, tl.int32, torch.int32,
         [0x01020304, 0x05060708, 0x11121314, 0x21222324], _pack_i32),
        ("f32", 4, tl.float32, torch.float32,
         [1.0, 2.0, 4.0, 8.0, 16.0], _pack_f32),
        ("bf16", 2, tl.bfloat16, torch.bfloat16,
         [1.0, 2.0, 3.0, 4.0, 5.0], _pack_bf16),
    ],
)
def test_widen_load_from_uint8_dtype_matrix(name, byte_width, target_dtype,
                                            torch_dtype, values, pack_fn):
    _require_npu0()

    block = 8
    base = 32
    host = torch.zeros(base + block * byte_width, dtype=torch.uint8)
    byte_offsets = [base + idx * byte_width for idx in range(len(values))]
    _write_packed_values(host, byte_offsets, values, pack_fn)

    src = host.npu()
    out = torch.empty((block,), device="npu", dtype=torch_dtype)

    _load_widen_from_u8[(1,)](src,
                              out,
                              len(values),
                              BASE=base,
                              BYTE_WIDTH=byte_width,
                              TARGET_DTYPE=target_dtype,
                              BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block,), dtype=torch_dtype)
    expected[:len(values)] = torch.tensor(values,
                                          dtype=torch.float32).to(torch_dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,byte_width,target_dtype,torch_dtype,values,pack_fn",
    [
        ("i16", 2, tl.int16, torch.int16, [101, 202, 303, 404], _pack_i16),
        ("i32", 4, tl.int32, torch.int32,
         [0x01010101, 0x02020202, 0x03030303], _pack_i32),
        ("f32", 4, tl.float32, torch.float32, [1.0, 2.0, 4.0], _pack_f32),
        ("bf16", 2, tl.bfloat16, torch.bfloat16,
         [1.0, 2.0, 3.0, 4.0], _pack_bf16),
    ],
)
def test_widen_store_to_uint8_dtype_matrix(name, byte_width, target_dtype,
                                           torch_dtype, values, pack_fn):
    _require_npu0()

    block = 8
    base = 16
    src = torch.tensor(values, dtype=torch.float32).to(torch_dtype).npu()
    dst = torch.zeros(base + block * byte_width, device="npu",
                      dtype=torch.uint8)

    _store_widen_to_u8[(1,)](src,
                             dst,
                             len(values),
                             BASE=base,
                             BYTE_WIDTH=byte_width,
                             TARGET_DTYPE=target_dtype,
                             BLOCK=block)
    _sync_npu()

    actual = dst.cpu()[base:base + len(values) * byte_width]
    expected = _expected_bytes(values, pack_fn)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_same_width_pointer_bitcast_i32_to_f32_is_unchanged():
    _require_npu0()

    block = 8
    word_base = 1
    values = [1.0, 2.0, 4.0, 8.0]
    raw = [0] + [_float_bits_to_i32(value) for value in values] + [0] * 4
    src = torch.tensor(raw, dtype=torch.int32).npu()
    out = torch.empty((block,), device="npu", dtype=torch.float32)

    _same_width_i32_pointer_bitcast[(1,)](src,
                                          out,
                                          word_base=word_base,
                                          n_elements=len(values),
                                          BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block,), dtype=torch.float32)
    expected[:len(values)] = torch.tensor(values, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_value_bitcast_i32_to_f32_is_unchanged():
    _require_npu0()

    block = 8
    values = [1.0, 2.0, 4.0, 8.0]
    raw = [_float_bits_to_i32(value) for value in values] + [0] * 4
    src = torch.tensor(raw, dtype=torch.int32).npu()
    out = torch.empty((block,), device="npu", dtype=torch.float32)

    _value_bitcast_i32_to_f32[(1,)](src,
                                    out,
                                    n_elements=len(values),
                                    BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block,), dtype=torch.float32)
    expected[:len(values)] = torch.tensor(values, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_narrow_i32_to_uint8_load():
    _require_npu0()

    block = 16
    word_base = 1
    words = [0, 0x01020304, 0x11121314, 0x21222324, 0x31323334]
    src = torch.tensor(words, dtype=torch.int32).npu()
    out = torch.empty((block,), device="npu", dtype=torch.uint8)

    _narrow_i32_to_u8_load[(1,)](src,
                                 out,
                                 word_base=word_base,
                                 n_bytes=12,
                                 BLOCK=block)
    _sync_npu()

    expected = _expected_bytes(words[1:4], _pack_i32)
    padded = torch.zeros((block,), dtype=torch.uint8)
    padded[:expected.numel()] = expected
    torch.testing.assert_close(out.cpu(), padded, rtol=0, atol=0)


def test_narrow_i32_tensor_pointers_to_uint8_load():
    _require_npu0()

    words = [
        0,
        0x01020304,
        0,
        0x11121314,
        0x21222324,
        0,
        0x31323334,
    ]
    word_offsets = [1, 3, 4, 6]
    src = torch.tensor(words, dtype=torch.int32).npu()
    offsets = torch.tensor(word_offsets, dtype=torch.int64).npu()
    out = torch.empty(
        (len(word_offsets),), device="npu", dtype=torch.uint8)

    _narrow_i32_tensor_ptrs_to_u8_load[(1,)](
        src, offsets, out, N=len(word_offsets), BLOCK=4)
    _sync_npu()

    expected = torch.tensor([0x04, 0x14, 0x24, 0x34], dtype=torch.uint8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_legacy_i1_pointer_to_uint8_load_is_unchanged():
    _require_npu0()

    values = [False, True, True, False, True, False, False, True]
    src = torch.tensor(values, dtype=torch.bool).npu()
    out = torch.empty((len(values),), device="npu", dtype=torch.uint8)

    _legacy_i1_pointer_to_u8_load[(1,)](
        src, out, N=len(values), BLOCK=8)
    _sync_npu()

    expected = torch.tensor(values, dtype=torch.uint8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_plain_uint8_load_store_is_unchanged():
    _require_npu0()

    block = 32
    n_elements = 19
    src_host = torch.arange(block, dtype=torch.uint8)
    src = src_host.npu()
    out = torch.empty((block,), device="npu", dtype=torch.uint8)

    _plain_u8_copy[(1,)](src, out, n_elements=n_elements, BLOCK=block)
    _sync_npu()

    expected = torch.zeros((block,), dtype=torch.uint8)
    expected[:n_elements] = src_host[:n_elements]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_misaligned_u8_to_i32_load_is_rejected():
    _require_npu0()

    src = torch.zeros((32,), device="npu", dtype=torch.uint8)
    out = torch.empty((), device="npu", dtype=torch.int32)

    with pytest.raises(Exception):
        _misaligned_u8_to_i32[(1,)](src, out)
        _sync_npu()
