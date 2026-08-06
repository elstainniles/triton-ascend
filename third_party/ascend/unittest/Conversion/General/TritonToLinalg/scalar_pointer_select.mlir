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

  func.func @pointer_cast_static_offset(%address: i64) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    %value = memref.load %view[%c0] : memref<1xf32, strided<[1], offset: 1>>
    return %value : f32
  }
}

// CHECK-LABEL: func.func @scalar_pointer_select(
// CHECK-SAME:  %[[LHS:[^ ,]+]]: memref<?xf32>, %[[RHS:[^ ,]+]]: memref<?xf32>
// CHECK:       %[[LHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[LHS]]
// CHECK:       %[[LHS_ADDRESS:.*]] = arith.index_cast %[[LHS_INDEX]] : index to i64
// CHECK:       %[[RHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[RHS]]
// CHECK:       %[[RHS_ADDRESS:.*]] = arith.index_cast %[[RHS_INDEX]] : index to i64
// CHECK:       %[[SELECTED:.*]] = arith.select %{{.*}}, %[[LHS_ADDRESS]], %[[RHS_ADDRESS]] : i64
// CHECK-NOT:   arith.select {{.*}} : memref
// CHECK-NOT:   hivm.hir.pointer_cast
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK:       return %[[SELECTED]] : i64

// CHECK-LABEL: func.func @scalar_pointer_roundtrip(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK-NOT:   hivm.hir.pointer_cast
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK:       return %[[ADDRESS]] : i64

// CHECK-LABEL: func.func @pointer_cast_static_offset(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[BYTE_WIDTH:.*]] = arith.constant 4 : i64
// CHECK:       %[[SIZE:.*]] = arith.constant 1 : index
// CHECK:       %[[OFFSET:.*]] = arith.constant 1 : i64
// CHECK:       %[[BYTE_OFFSET:.*]] = arith.muli %[[OFFSET]], %[[BYTE_WIDTH]] : i64
// CHECK:       %[[REAL_ADDRESS:.*]] = arith.addi %[[ADDRESS]], %[[BYTE_OFFSET]] : i64
// CHECK:       %[[REBASED_POINTER:.*]] = hivm.hir.pointer_cast(%[[REAL_ADDRESS]]) [%[[SIZE]]] : memref<?xf32>
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[REBASED_POINTER]] to offset: [0], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1]>>
// CHECK:       memref.load %[[VIEW]][%{{.*}}] : memref<1xf32, strided<[1]>>
