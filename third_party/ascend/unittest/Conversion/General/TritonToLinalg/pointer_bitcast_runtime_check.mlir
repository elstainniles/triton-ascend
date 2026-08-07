// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK: func.func private @triton_assert{{.*}}(i1) attributes {msg = "pointer bitcast offset is not divisible by 2"}
// CHECK-LABEL: func.func @runtime_check_scalar_offset
// CHECK-DAG: %[[RATIO:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[REMAINDER:.*]] = arith.remsi %{{.*}}, %[[RATIO]] : i32
// CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REMAINDER]], %[[ZERO]] : i32
// CHECK: call @triton_assert{{.*}}(%[[ALIGNED]]) : (i1) -> ()
// CHECK: arith.divsi
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @runtime_check_scalar_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %byte_offset: i32) attributes {noinline = false} {
    %byte_ptr = tt.addptr %src, %byte_offset : !tt.ptr<i8>, i32
    %bf16_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<bf16>
    %value = tt.load %bf16_ptr : !tt.ptr<bf16>
    tt.store %dst, %value : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// CHECK: func.func private @triton_assert{{.*}}(tensor<4xi1>) attributes {msg = "pointer bitcast offset is not divisible by 2"}
// CHECK-LABEL: func.func @runtime_check_tensor_offset
// CHECK: %[[REMAINDER:.*]] = arith.remsi {{.*}} : tensor<4xi32>
// CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REMAINDER]], {{.*}} : tensor<4xi32>
// CHECK: call @triton_assert{{.*}}(%[[ALIGNED]]) : (tensor<4xi1>) -> ()
// CHECK: arith.divsi {{.*}} : index
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @runtime_check_tensor_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %offsets_src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %offset_bases = tt.splat %offsets_src : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %offset_ptrs = tt.addptr %offset_bases, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %byte_offsets = tt.load %offset_ptrs : tensor<4x!tt.ptr<i32>>
    %srcs = tt.splat %src : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %byte_ptrs = tt.addptr %srcs, %byte_offsets : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %bf16_ptrs = tt.bitcast %byte_ptrs : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<bf16>>
    %values = tt.load %bf16_ptrs : tensor<4x!tt.ptr<bf16>>
    %dsts = tt.splat %dst : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<4x!tt.ptr<bf16>>, tensor<4xi32>
    tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @static_divisible_offset
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.cmpi
// CHECK-NOT: triton_assert
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.cmpi
// CHECK-NOT: triton_assert
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @static_divisible_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %byte_offset = arith.constant 2 : i32
    %byte_ptr = tt.addptr %src, %byte_offset : !tt.ptr<i8>, i32
    %bf16_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<bf16>
    %value = tt.load %bf16_ptr : !tt.ptr<bf16>
    tt.store %dst, %value : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @runtime_check_unstructured_tensor_offset
// CHECK: call @triton_assert{{.*}}(%{{.*}}) : (tensor<4xi1>) -> ()
// CHECK: scf.for
// CHECK: arith.divsi {{.*}} : i64
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @runtime_check_unstructured_tensor_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %offsets_src: !tt.ptr<i64> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i64 = arith.constant 4 : i64
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ratio = arith.constant dense<4> : tensor<4xi64>
    %limit = arith.constant dense<4> : tensor<4xi32>
    %zero_i64 = arith.constant dense<0> : tensor<4xi64>
    %zero_i32 = arith.constant dense<0> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %offset_bases = tt.splat %offsets_src : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %offset_ptrs = tt.addptr %offset_bases, %range : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %offsets = tt.load %offset_ptrs, %mask, %zero_i64 : tensor<4x!tt.ptr<i64>>
    %typed_src = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
    %remainder = arith.remsi %offsets, %ratio : tensor<4xi64>
    %aligned = arith.cmpi eq, %remainder, %zero_i64 : tensor<4xi64>
    tt.assert %aligned, "pointer bitcast offset is not divisible by 4" : tensor<4xi1>
    %empty = tensor.empty() : tensor<4xi32>
    %loaded = scf.for %index = %c0 to %c4 step %c1 iter_args(%result = %empty) -> (tensor<4xi32>) {
      %offset = tensor.extract %offsets[%index] {DiscreteMemAccess} : tensor<4xi64>
      %scaled = arith.divsi %offset, %c4_i64 : i64
      %ptr = tt.addptr %typed_src, %scaled : !tt.ptr<i32>, i64
      %value = tt.load %ptr {DiscreteMemAccess} : !tt.ptr<i32>
      %next = tensor.insert %value into %result[%index] : tensor<4xi32>
      scf.yield {DiscreteMemAccess} %next : tensor<4xi32>
    } {ExtractedLoadOrStore}
    %selected = arith.select %mask, %loaded, %zero_i32 {DiscreteMemAccess} : tensor<4xi1>, tensor<4xi32>
    %dst_bases = tt.splat %dst : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dst_bases, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %dst_ptrs, %selected, %mask : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @value_bitcast_is_unchanged
// CHECK: arith.bitcast {{.*}} : tensor<4xi32> to tensor<4xf32>
// CHECK-NOT: triton_assert
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @value_bitcast_is_unchanged(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %src_bases = tt.splat %src : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_bases, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %values = tt.load %src_ptrs : tensor<4x!tt.ptr<i32>>
    %bits = tt.bitcast %values : tensor<4xi32> -> tensor<4xf32>
    %dst_bases = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_bases, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %dst_ptrs, %bits : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
