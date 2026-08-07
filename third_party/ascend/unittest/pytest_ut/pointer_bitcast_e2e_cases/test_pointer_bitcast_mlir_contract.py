import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def _find_triton_opt():
    configured = os.environ.get("TRITON_OPT")
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)
        pytest.fail(f"TRITON_OPT does not point to a file: {configured}")
    discovered = shutil.which("triton-opt")
    if discovered:
        return discovered
    pytest.fail("triton-opt is required; add it to PATH or set TRITON_OPT")


def _run_ttolinalg(tmp_path, name, module):
    source = tmp_path / f"{name}.mlir"
    source.write_text(textwrap.dedent(module), encoding="utf-8")
    return subprocess.run(
        [_find_triton_opt(), "--triton-to-linalg=named-ops=True", str(source)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_dynamic_ratio4_inserts_runtime_check_before_division(tmp_path):
    result = _run_ttolinalg(tmp_path, "dynamic_ratio4", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @dynamic_ratio4(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %byte_offset: i32) attributes {noinline = false} {
            %byte_ptr = tt.addptr %src, %byte_offset : !tt.ptr<i8>, i32
            %wide_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    remainder = output.index("arith.remsi")
    assertion = output.index("call @triton_assert")
    division = output.index("arith.divsi")
    assert remainder < assertion < division, output
    assert "pointer bitcast offset is not divisible by 4" in output


def test_axisinfo_proven_ratio4_does_not_insert_runtime_check(tmp_path):
    result = _run_ttolinalg(tmp_path, "axisinfo_ratio4", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @axisinfo_ratio4(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %offsets = arith.muli %range, %four : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %srcs, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %values = tt.load %wide_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout
    assert "builtin.unrealized_conversion_cast" not in result.stdout
    assert "hivm.hir.pointer_cast" in result.stdout


def test_static_tensor_with_one_indivisible_lane_is_rejected(tmp_path):
    result = _run_ttolinalg(tmp_path, "static_tensor_invalid", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @static_tensor_invalid(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %offsets = arith.constant dense<[0, 4, 6, 12]>
                : tensor<4xi32>
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %srcs, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %values = tt.load %wide_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0, diagnostics
    assert "statically known not to be divisible" in diagnostics
    assert "divisible by 4" in diagnostics


def test_same_width_pointer_bitcast_has_no_alignment_check(tmp_path):
    result = _run_ttolinalg(tmp_path, "same_width", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @same_width(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
              %offset: i32) attributes {noinline = false} {
            %source_ptr = tt.addptr %src, %offset : !tt.ptr<i32>, i32
            %float_ptr = tt.bitcast %source_ptr
                : !tt.ptr<i32> -> !tt.ptr<f32>
            %value = tt.load %float_ptr : !tt.ptr<f32>
            tt.store %dst, %value : !tt.ptr<f32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_non_integral_pointee_width_ratio_is_rejected(tmp_path):
    result = _run_ttolinalg(tmp_path, "non_integral_ratio", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @non_integral_ratio(
              %src: !tt.ptr<i24> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i16> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i24> -> !tt.ptr<i16>
            %value = tt.load %wide_ptr : !tt.ptr<i16>
            tt.store %dst, %value : !tt.ptr<i16>
            tt.return
          }
        }
    """)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0, diagnostics
    assert "integral ratio" in diagnostics


def test_direct_function_argument_different_width_bitcast(tmp_path):
    result = _run_ttolinalg(tmp_path, "direct_function_argument", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @direct_function_argument(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout
    assert "arith.divsi" not in result.stdout, result.stdout


def test_splat_tensor_pointer_bitcast_is_canonicalized(tmp_path):
    result = _run_ttolinalg(tmp_path, "splat_tensor_pointer", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @splat_tensor_pointer(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %wide_ptrs = tt.bitcast %srcs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %load_offsets = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %load_ptrs = tt.addptr %wide_ptrs, %load_offsets
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %values = tt.load %load_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %store_ptrs = tt.addptr %dsts, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %store_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_nested_pointer_bitcasts_are_fused(tmp_path):
    result = _run_ttolinalg(tmp_path, "nested_pointer_bitcasts", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @nested_pointer_bitcasts(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %half_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i16>
            %word_ptr = tt.bitcast %half_ptr : !tt.ptr<i16> -> !tt.ptr<i32>
            %value = tt.load %word_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_legacy_i1_to_i8_pointer_bitcast_is_unchanged(tmp_path):
    result = _run_ttolinalg(tmp_path, "legacy_i1_to_i8", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @legacy_i1_to_i8(
              %src: !tt.ptr<i1> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %byte_ptr = tt.bitcast %src : !tt.ptr<i1> -> !tt.ptr<i8>
            %value = tt.load %byte_ptr : !tt.ptr<i8>
            tt.store %dst, %value : !tt.ptr<i8>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_address_space_mismatch_is_rejected(tmp_path):
    result = _run_ttolinalg(tmp_path, "address_space_mismatch", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @address_space_mismatch(
              %src: !tt.ptr<i8, 0> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8, 0> -> !tt.ptr<i32, 1>
            %value = tt.load %wide_ptr : !tt.ptr<i32, 1>
            tt.store %dst, %value : !tt.ptr<i32, 1>
            tt.return
          }
        }
    """)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0, diagnostics
    assert "different address spaces" in diagnostics


def test_uncanonicalized_tensor_pointer_argument_is_rejected(tmp_path):
    result = _run_ttolinalg(tmp_path, "uncanonicalized_tensor_argument", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @uncanonicalized_tensor_argument(
              %srcs: tensor<4x!tt.ptr<i8>>,
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptrs = tt.bitcast %srcs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %load_ptrs = tt.addptr %wide_ptrs, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %values = tt.load %load_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %store_ptrs = tt.addptr %dsts, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %store_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0, diagnostics
    assert "not canonicalized to a scalar base pointer" in diagnostics
