// RUN: triton-opt --triton-to-linalg %s -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @scalar_pointer_select(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %condition: i1) -> i64 {
    %selected = arith.select %condition, %lhs, %rhs : !tt.ptr<f32>
    %address = tt.ptr_to_int %selected : !tt.ptr<f32> -> i64
    tt.return %address : i64
  }

  tt.func public @scalar_pointer_roundtrip(%address: i64) -> i64 {
    %pointer = tt.int_to_ptr %address : i64 -> !tt.ptr<f32>
    %roundtrip = tt.ptr_to_int %pointer : !tt.ptr<f32> -> i64
    tt.return %roundtrip : i64
  }
}

// CHECK-LABEL: func.func @scalar_pointer_select(
// CHECK-SAME:  %[[LHS:.*]]: memref<?xf32>, %[[RHS:.*]]: memref<?xf32>
// CHECK:       %[[LHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[LHS]]
// CHECK:       %[[LHS_ADDRESS:.*]] = arith.index_cast %[[LHS_INDEX]] : index to i64
// CHECK:       %[[RHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[RHS]]
// CHECK:       %[[RHS_ADDRESS:.*]] = arith.index_cast %[[RHS_INDEX]] : index to i64
// CHECK:       %[[SELECTED:.*]] = arith.select %{{.*}}, %[[LHS_ADDRESS]], %[[RHS_ADDRESS]] : i64
// CHECK-NOT:   arith.select {{.*}} : memref
// CHECK:       %[[POINTER:.*]] = hivm.pointer_cast
// CHECK-SAME:  %[[SELECTED]]
// CHECK-SAME:  memref<?xf32>
// CHECK:       %[[POINTER_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[POINTER]]
// CHECK:       return %{{.*}} : i64

// CHECK-LABEL: func.func @scalar_pointer_roundtrip(
// CHECK-SAME:  %[[ADDRESS:.*]]: i64
// CHECK:       %[[POINTER:.*]] = hivm.pointer_cast
// CHECK-SAME:  %[[ADDRESS]]
// CHECK-SAME:  memref<?xf32>
// CHECK:       %[[POINTER_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[POINTER]]
// CHECK:       %[[ROUNDTRIP:.*]] = arith.index_cast %[[POINTER_INDEX]] : index to i64
// CHECK:       return %[[ROUNDTRIP]] : i64
