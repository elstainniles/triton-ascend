// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"

namespace mlir::triton::cfg {

BatchResult<std::optional<BatchLogicalAxis>>
isSafeToBatchLoop(const BatchLoopFact *programLoop, int gridAxis) {
  // The caller supplies gate 0's form check from BatchLoopForm.h. Having no
  // program-axis loop is valid: straight-line kernels and kernels with only
  // inner loops retain program-level widening without changing their loops.
  using Result = BatchResult<std::optional<BatchLogicalAxis>>;
  if (!programLoop)
    return Result::success(std::nullopt);

  // Gate 1: normalize loop coordinates.
  auto schedule = normalizeBatchLoopCoordinates(*programLoop, gridAxis);
  if (!schedule.hasValue())
    return Result::decline(schedule.diagnostic());

  // Both distributions retain each original program's rows: lane u represents
  // g*F+u with unchanged row order and writeback slots. Recurrences and memory
  // dependencies within that program therefore retain their original order.
  //
  // The old LOOP_REDUCTION_CROSSES_ROWS and LOOP_DISTANCE_UNPROVEN checks
  // assumed an unfolded schedule that split one program's segment into groups
  // of F rows, multiplied the step by F and kept the grid unchanged. Here the
  // grid is divided by F and each lane retains an entire original segment.
  //
  // Gate 1 instead requires provable per-lane segment bounds for blocked
  // distribution; otherwise it returns UNPROVEN_LANE_SEGMENT_BOUND.
  return Result::success(std::move(schedule).takeValue());
}

} // namespace mlir::triton::cfg
