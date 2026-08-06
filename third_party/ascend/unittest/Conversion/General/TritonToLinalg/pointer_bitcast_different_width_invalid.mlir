// RUN: not triton-opt --triton-to-linalg="named-ops=True" %s 2>&1 | FileCheck %s

// CHECK: cannot reinterpret ptr<i8> as ptr<i32>: pre-cast offset must be divisible by 4 for every element; alignment could not be proven
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @reject_unproven_dynamic_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %byte_offset: i32) attributes {noinline = false} {
    %byte_ptr = tt.addptr %src, %byte_offset : !tt.ptr<i8>, i32
    %i32_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i32>
    %value = tt.load %i32_ptr : !tt.ptr<i32>
    tt.store %dst, %value : !tt.ptr<i32>
    tt.return
  }
}
