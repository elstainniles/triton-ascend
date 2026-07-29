/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;
using mlir::triton::controlflow::ControlFlowOpAnalysis;
using mlir::triton::controlflow::ControlFlowRewriteContext;
using mlir::triton::controlflow::ControlFlowRewritePlan;
using mlir::triton::controlflow::ControlFlowRewritePolicy;
using mlir::triton::controlflow::ControlFlowSlotAnalysis;
using mlir::triton::controlflow::DecomposedValue;

namespace mlir::triton::controlflow {

Value ControlFlowRewriteContext::remap(Value value) const {
  if (Value mapped = valueMapping.lookupOrNull(value))
    return mapped;
  return value;
}

const DecomposedValue *ControlFlowRewriteContext::lookup(Value value) const {
  auto it = decomposedValues.find(value);
  return it == decomposedValues.end() ? nullptr : &it->second;
}

} // namespace mlir::triton::controlflow

namespace {

// Keep the mechanical if/for/while rewrite in one translation unit. These
// handlers are mutually recursive through rewriteBodyOps(), share one
// short-lived RewriteEnv, and must agree on signature expansion, nested-op
// ordering and failure cleanup. Splitting them by op kind would expose those
// private invariants through additional internal headers without creating an
// independently reusable component.
//
//===----------------------------------------------------------------------===//
// Per-rewrite state and generic component manipulation
//===----------------------------------------------------------------------===//

struct RewriteEnv {
  RewriteEnv(const ControlFlowRewritePolicy &policy,
             const ControlFlowRewritePlan &plan)
      : policy(policy), plan(plan) {}

  // Maps values from the original region to values in the replacement region.
  IRMapping valueMapping;
  // Concrete component state keyed by original values. Keeping this alongside
  // the mapping lets pointer producers be flattened across nested rewrites.
  DenseMap<Value, DecomposedValue> decomposedValues;
  const ControlFlowRewritePolicy &policy;
  const ControlFlowRewritePlan &plan;
};

// RewriteEnv is copied when entering a newly built region. The copy inherits
// mappings visible at the region boundary and records additional mappings only
// for that recursive rewrite. Nothing is stored on the IR or shared between
// decomposition policies.

struct LoopPointerInfo {
  // Original iter-argument/result position before signature expansion.
  unsigned oldIndex = 0;
  // Concrete descriptor used as the reconstruction template.
  DecomposedValue initInfo;
  // Ordered schema decided by ControlFlowSlotAnalysis.
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  // Positions occupied by those components in the replacement operation.
  SmallVector<unsigned> newIndices;
};

struct IfPointerInfo {
  unsigned oldIndex = 0;
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  std::optional<DecomposedValue> thenInfo;
};

static Value remapValue(Value value, const RewriteEnv &env) {
  return ControlFlowRewriteContext(env.valueMapping, env.decomposedValues)
      .remap(value);
}

static FailureOr<DecomposedValue> decompose(Value value, const RewriteEnv &env,
                                            OpBuilder &builder, Location loc) {
  return env.policy.decompose(
      value, ControlFlowRewriteContext(env.valueMapping, env.decomposedValues),
      builder, loc);
}

static void recordDecomposition(Value oldValue, const DecomposedValue &info,
                                Value rebuiltValue, RewriteEnv &env) {
  env.decomposedValues[oldValue] = info;
  env.valueMapping.map(oldValue, rebuiltValue);
}

static SmallVector<Value> getComponentValues(const DecomposedValue &info,
                                             ArrayRef<unsigned> indices) {
  // Callers validate the policy-provided indices before reaching this helper.
  // Preserve their order because it is also the replacement signature order.
  SmallVector<Value> values;
  values.reserve(indices.size());
  for (unsigned index : indices)
    values.push_back(info.components[index]);
  return values;
}

static FailureOr<DecomposedValue>
withComponentValues(DecomposedValue info, ArrayRef<unsigned> indices,
                    ArrayRef<Value> values) {
  // Start from one branch/init descriptor and replace only the components that
  // crossed the boundary. Invariants never enter the generic SCF signature.
  if (indices.size() != values.size())
    return failure();
  for (auto [index, value] : llvm::zip(indices, values)) {
    if (index >= info.components.size() ||
        info.components[index].getType() != value.getType())
      return failure();
    info.components[index] = value;
  }
  return info;
}

static LogicalResult castPlannedComponents(DecomposedValue &value,
                                           ArrayRef<unsigned> componentIndices,
                                           ArrayRef<Type> componentTypes,
                                           OpBuilder &builder, Location loc) {
  if (componentIndices.size() != componentTypes.size())
    return failure();
  for (auto [index, type] : llvm::zip(componentIndices, componentTypes)) {
    if (index >= value.components.size())
      return failure();
    FailureOr<Value> component =
        castIntegerLike(builder, loc, value.components[index], type);
    if (failed(component))
      return failure();
    value.components[index] = *component;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shared recursive body rewrite
//===----------------------------------------------------------------------===//

static LoopPointerInfo *findLoopInfo(SmallVectorImpl<LoopPointerInfo> &infos,
                                     unsigned oldIndex) {
  for (LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const LoopPointerInfo *findLoopInfo(ArrayRef<LoopPointerInfo> infos,
                                           unsigned oldIndex) {
  for (const LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static SmallVector<Value> collectForComponents(const LoopPointerInfo &info,
                                               scf::ForOp forOp,
                                               bool useResults) {
  // `newIndices` is shared by init operands, region arguments and results of an
  // scf.for, so choosing the source is sufficient to recover the descriptor.
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices)
    values.push_back(useResults ? forOp.getResult(newIndex)
                                : forOp.getRegionIterArgs()[newIndex]);
  return values;
}

static SmallVector<Value> collectWhileComponents(const LoopPointerInfo &info,
                                                 scf::WhileOp whileOp,
                                                 bool useResults,
                                                 bool useAfterArgs) {
  // scf.while has two region-argument lists in addition to its results; all
  // three use the same expanded positional schema.
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices) {
    if (useResults)
      values.push_back(whileOp.getResult(newIndex));
    else if (useAfterArgs)
      values.push_back(whileOp.getAfterArguments()[newIndex]);
    else
      values.push_back(whileOp.getBeforeArguments()[newIndex]);
  }
  return values;
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env);

static LogicalResult materializePointerResult(Operation &originalOp,
                                              Operation *clonedOp,
                                              OpBuilder &builder,
                                              RewriteEnv &env) {
  // Each policy decides which pointer-producing operations need their exact
  // components recorded immediately after cloning.
  if (!env.policy.shouldDecomposeOperation(&originalOp))
    return success();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(clonedOp);

  bool decomposedAllResults = clonedOp->getNumResults() != 0;
  for (auto [oldResult, clonedResult] :
       llvm::zip(originalOp.getResults(), clonedOp->getResults())) {
    if (!env.policy.isDecompositionTarget(oldResult)) {
      decomposedAllResults = false;
      continue;
    }

    FailureOr<DecomposedValue> info =
        decompose(clonedResult, env, builder, oldResult.getLoc());
    if (failed(info))
      return failure();

    Value rebuilt = env.policy.recompose(*info, builder, oldResult.getLoc());
    if (!rebuilt)
      return failure();
    recordDecomposition(oldResult, *info, rebuilt, env);
  }

  // The replacement is recorded in the SSA mapping, so a side-effect-free
  // clone whose every result was decomposed is redundant once it has no users.
  if (decomposedAllResults && clonedOp->use_empty() &&
      isMemoryEffectFree(clonedOp))
    clonedOp->erase();

  return success();
}

static LogicalResult rewriteBodyOps(Block *oldBlock, OpBuilder &builder,
                                    RewriteEnv &env) {
  // Process operations in program order. Nested control flow is rewritten
  // recursively with the same policy; ordinary operations are cloned through
  // the current SSA mapping.
  for (Operation &originalOp : oldBlock->without_terminator()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(originalOp)) {
      const ControlFlowOpAnalysis *analysis = env.plan.lookup(&originalOp);
      if (!analysis)
        return failure();
      if (analysis->needsRewrite()) {
        if (failed(rewriteControlFlowOp(&originalOp, builder, env)))
          return failure();
        continue;
      }
    }
    Operation *clonedOp = builder.clone(originalOp, env.valueMapping);
    if (failed(materializePointerResult(originalOp, clonedOp, builder, env)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.for rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteForOp(scf::ForOp forOp, OpBuilder &builder,
                                  RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(forOp);
  if (!analysis || !analysis->needsRewrite())
    return failure();
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // The read-only analysis has already fixed every expanded slot and type.
  // Materialization here recovers only the concrete values for that schema.
  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    unsigned idx = slot.oldIndex;
    if (idx >= forOp.getInitArgs().size() || idx >= yieldOp.getNumOperands() ||
        !env.policy.matches(forOp.getRegionIterArgs()[idx].getType()))
      return failure();

    FailureOr<DecomposedValue> initInfo =
        decompose(forOp.getInitArgs()[idx], env, builder, forOp.getLoc());
    if (failed(initInfo) || failed(castPlannedComponents(
                                *initInfo, slot.componentIndices,
                                slot.componentTypes, builder, forOp.getLoc())))
      return failure();
    pointerInfos.push_back(LoopPointerInfo{
        idx, *initInfo, slot.componentIndices, slot.componentTypes, {}});
  }

  SmallVector<Value> newInitArgs;
  SmallVector<unsigned> oldToNewStart(forOp.getInitArgs().size(), 0);
  // Expand each owned pointer init into its runtime components. Non-pointer and
  // other-policy slots retain one position in the new signature.
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    oldToNewStart[idx] = newInitArgs.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component :
           getComponentValues(info->initInfo, info->componentIndices)) {
        info->newIndices.push_back(newInitArgs.size());
        newInitArgs.push_back(component);
      }
      continue;
    }
    newInitArgs.push_back(remapValue(initArg, env));
  }

  bool bodyOk = true;
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), remapValue(forOp.getLowerBound(), env),
      remapValue(forOp.getUpperBound(), env), remapValue(forOp.getStep(), env),
      newInitArgs,
      [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange args) {
        RewriteEnv bodyEnv = env;
        bodyEnv.valueMapping.map(forOp.getInductionVar(), iv);

        // Reconstruct the original iter-argument type at region entry before
        // cloning users. This keeps the body semantics independent of the
        // signature expansion.
        for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, bodyEnv);
            continue;
          }
          bodyEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(forOp.getBody(), bodyBuilder, bodyEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        // Decompose yielded pointers back into the component order selected by
        // the read-only loop analysis.
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> nextInfo =
                decompose(oldOperand, bodyEnv, bodyBuilder, yieldOp.getLoc());
            if (failed(nextInfo) ||
                failed(castPlannedComponents(*nextInfo, info->componentIndices,
                                             info->componentTypes, bodyBuilder,
                                             yieldOp.getLoc()))) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*nextInfo, info->componentIndices))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, bodyEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newForOp->setAttrs(forOp->getAttrs());

  if (!bodyOk) {
    newForOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newForOp);
  // Rebuild pointer results after the new loop and map every untouched result
  // to the corresponding expanded-result index.
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          info->initInfo, info->componentIndices,
          collectForComponents(*info, newForOp, /*useResults=*/true));
      if (failed(resultInfo)) {
        newForOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newForOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newForOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scf.while rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteWhileOp(scf::WhileOp whileOp, OpBuilder &builder,
                                    RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(whileOp);
  if (!analysis || !analysis->needsRewrite())
    return failure();
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // The before arguments, condition forwarded values, after arguments and
  // yield operands all consume the same precomputed positional schema.
  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    unsigned idx = slot.oldIndex;
    if (idx >= whileOp.getBeforeArguments().size() ||
        !env.policy.matches(whileOp.getBeforeArguments()[idx].getType()) ||
        idx >= whileOp.getInits().size() ||
        idx >= conditionOp.getArgs().size() || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<DecomposedValue> initInfo =
        decompose(whileOp.getInits()[idx], env, builder, whileOp.getLoc());
    if (failed(initInfo) ||
        failed(castPlannedComponents(*initInfo, slot.componentIndices,
                                     slot.componentTypes, builder,
                                     whileOp.getLoc())))
      return failure();
    pointerInfos.push_back(LoopPointerInfo{
        idx, *initInfo, slot.componentIndices, slot.componentTypes, {}});
  }

  // Expand inits and result types in lockstep. oldToNewStart keeps untouched
  // positions addressable even when earlier pointer slots expand by rank.
  SmallVector<Value> newInits;
  SmallVector<Type> newResultTypes;
  SmallVector<unsigned> oldToNewStart(whileOp.getInits().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(whileOp.getInits())) {
    oldToNewStart[idx] = newInits.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component :
           getComponentValues(info->initInfo, info->componentIndices)) {
        info->newIndices.push_back(newInits.size());
        newInits.push_back(component);
        newResultTypes.push_back(component.getType());
      }
      continue;
    }
    newInits.push_back(remapValue(initArg, env));
    newResultTypes.push_back(whileOp.getResult(idx).getType());
  }

  bool bodyOk = true;
  auto newWhileOp = builder.create<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv beforeEnv = env;
        // Rebuild pointer values at the before-region boundary, rewrite the
        // body, then decompose the values forwarded by scf.condition.
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getBeforeArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, beforeEnv);
            continue;
          }
          beforeEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(whileOp.getBeforeBody(), bodyBuilder,
                                  beforeEnv)))
          bodyOk = false;

        SmallVector<Value> newConditionArgs;
        for (auto [idx, oldArg] : llvm::enumerate(conditionOp.getArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> conditionInfo =
                decompose(oldArg, beforeEnv, bodyBuilder, conditionOp.getLoc());
            if (failed(conditionInfo) ||
                failed(castPlannedComponents(
                    *conditionInfo, info->componentIndices,
                    info->componentTypes, bodyBuilder, conditionOp.getLoc()))) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newConditionArgs.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*conditionInfo, info->componentIndices))
              newConditionArgs.push_back(component);
            continue;
          }
          newConditionArgs.push_back(remapValue(oldArg, beforeEnv));
        }

        bodyBuilder.create<scf::ConditionOp>(
            conditionOp.getLoc(),
            remapValue(conditionOp.getCondition(), beforeEnv),
            newConditionArgs);
      },
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv afterEnv = env;
        // Apply the same reconstruction/decomposition contract to the after
        // region and its backedge yield.
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getAfterArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, afterEnv);
            continue;
          }
          afterEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(
                rewriteBodyOps(whileOp.getAfterBody(), bodyBuilder, afterEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> nextInfo =
                decompose(oldOperand, afterEnv, bodyBuilder, yieldOp.getLoc());
            if (failed(nextInfo) ||
                failed(castPlannedComponents(*nextInfo, info->componentIndices,
                                             info->componentTypes, bodyBuilder,
                                             yieldOp.getLoc()))) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*nextInfo, info->componentIndices))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, afterEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newWhileOp->setAttrs(whileOp->getAttrs());

  if (!bodyOk) {
    newWhileOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newWhileOp);
  for (auto [idx, oldResult] : llvm::enumerate(whileOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          info->initInfo, info->componentIndices,
          collectWhileComponents(*info, newWhileOp, /*useResults=*/true,
                                 /*useAfterArgs=*/false));
      if (failed(resultInfo)) {
        newWhileOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newWhileOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newWhileOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scf.if component planning and rewrite
//===----------------------------------------------------------------------===//

static IfPointerInfo *findIfInfo(SmallVectorImpl<IfPointerInfo> &infos,
                                 unsigned oldIndex) {
  for (IfPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const IfPointerInfo *findIfInfo(ArrayRef<IfPointerInfo> infos,
                                       unsigned oldIndex) {
  for (const IfPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static LogicalResult rewriteIfOp(scf::IfOp ifOp, OpBuilder &builder,
                                 RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(ifOp);
  if (!analysis || !analysis->needsRewrite() ||
      (!ifOp.elseBlock() && analysis->rewritesOwnSignature()))
    return failure();

  bool hasElse = static_cast<bool>(ifOp.elseBlock());
  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = hasElse ? ifOp.elseYield() : scf::YieldOp();
  SmallVector<IfPointerInfo, 4> pointerInfos;

  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    if (slot.oldIndex >= ifOp.getNumResults() ||
        !env.policy.matches(ifOp.getResult(slot.oldIndex).getType()) ||
        slot.componentIndices.size() != slot.componentTypes.size())
      return failure();
    pointerInfos.push_back(IfPointerInfo{slot.oldIndex, slot.componentIndices,
                                         slot.componentTypes, std::nullopt});
  }

  // Expand only result positions selected by analysis. An if with no pointer
  // result may still be rebuilt because one of its nested operations changes.
  SmallVector<Type> newResultTypes;
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      newResultTypes.append(info->componentTypes.begin(),
                            info->componentTypes.end());
      continue;
    }
    newResultTypes.push_back(result.getType());
  }

  bool bodyOk = true;
  auto buildBranch = [&](OpBuilder &branchBuilder,
                         bool isThen) -> LogicalResult {
    // Each branch gets an isolated environment because values defined in one
    // branch must never be visible while cloning the other branch.
    RewriteEnv branchEnv = env;
    Block *oldBlock = isThen ? ifOp.thenBlock() : ifOp.elseBlock();
    scf::YieldOp oldYield = isThen ? thenYield : elseYield;
    if (failed(rewriteBodyOps(oldBlock, branchBuilder, branchEnv)))
      return failure();

    SmallVector<Value> newYieldOperands;
    for (auto [idx, oldOperand] : llvm::enumerate(oldYield.getOperands())) {
      if (IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
        FailureOr<DecomposedValue> branchInfo =
            decompose(oldOperand, branchEnv, branchBuilder, oldYield.getLoc());
        if (failed(branchInfo) ||
            failed(castPlannedComponents(*branchInfo, info->componentIndices,
                                         info->componentTypes, branchBuilder,
                                         oldYield.getLoc())))
          return failure();
        if (isThen)
          info->thenInfo = *branchInfo;
        SmallVector<Value> values =
            getComponentValues(*branchInfo, info->componentIndices);
        newYieldOperands.append(values.begin(), values.end());
        continue;
      }
      newYieldOperands.push_back(remapValue(oldOperand, branchEnv));
    }
    branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
    return success();
  };

  // Create the shell first, then clone each old branch into the corresponding
  // new region with independent mappings.
  auto newIfOp =
      builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                remapValue(ifOp.getCondition(), env), hasElse);
  newIfOp->setAttrs(ifOp->getAttrs());

  // The then region always exists, including for a result-less one-arm if.
  // Rewriting it is still required when it contains affected nested SCF.
  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.thenBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.thenBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.thenBlock());
    }
    if (failed(buildBranch(builder, /*isThen=*/true)))
      bodyOk = false;
  }
  // An else region exists only for the two-arm form. In particular, do not
  // access elseBlock() merely because the then region contains nested work.
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.elseBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.elseBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    if (failed(buildBranch(builder, /*isThen=*/false)))
      bodyOk = false;
  }

  if (!bodyOk) {
    newIfOp.erase();
    return failure();
  }

  // Reassemble the pointer immediately after the replacement if. Downstream
  // operations therefore keep their original operand types; decomposition is
  // limited to the control-flow boundary itself.
  builder.setInsertionPointAfter(newIfOp);
  unsigned newResultIndex = 0;
  for (auto [idx, oldResult] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      SmallVector<Value> componentValues;
      for (unsigned i = 0; i < info->componentIndices.size(); ++i)
        componentValues.push_back(newIfOp.getResult(newResultIndex++));
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          *info->thenInfo, info->componentIndices, componentValues);
      if (failed(resultInfo)) {
        newIfOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newIfOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newIfOp.getResult(newResultIndex++));
  }

  return success();
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env) {
  // Keep the operation dispatch next to the shared recursive implementation:
  // all supported region operations must obey the same mapping and cleanup
  // rules. Pointer-specific semantics enter only through env.policy.
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp, builder, env);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp, builder, env);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp, builder, env);
  // TODO: Add the frontend-produced scope.scope operation here. Scope support
  // belongs in this shared plumbing rather than in both decompositions.
  return failure();
}

static FailureOr<SmallVector<Value>>
collectReplacements(Operation *op, const RewriteEnv &env) {
  SmallVector<Value> replacements;
  replacements.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    // Unlike remapValue(), replacement collection must not fall back to the
    // original result. Such a fallback would hide an unhandled result slot and
    // ask replaceOp to replace a value with itself.
    Value replacement = env.valueMapping.lookupOrNull(result);
    if (!replacement)
      return failure();
    replacements.push_back(replacement);
  }
  return replacements;
}

static LogicalResult
tryDecoupleControlFlowOp(Operation *op, IRRewriter &rewriter,
                         const ControlFlowRewritePolicy &policy,
                         const ControlFlowRewritePlan &plan) {
  // Build a replacement beside the original operation. The original operation
  // itself remains until every result has a valid mapped value, after which the
  // standard rewriter performs the externally visible replacement.
  // TODO: Track and erase policy materializations created outside the new SCF
  // operation if an unexpected rewrite-time validation fails. Read-only
  // analysis makes that path exceptional, but failure should still be atomic.
  RewriteEnv env(policy, plan);
  rewriter.setInsertionPoint(op);
  if (failed(rewriteControlFlowOp(op, rewriter, env)))
    return failure();

  FailureOr<SmallVector<Value>> replacements = collectReplacements(op, env);
  if (failed(replacements))
    return failure();
  rewriter.replaceOp(op, *replacements);
  return success();
}

} // namespace

namespace mlir::triton::controlflow {

LogicalResult
applyControlFlowRewritePlan(ModuleOp module,
                            const ControlFlowRewritePolicy &policy,
                            const ControlFlowRewritePlan &plan) {
  IRRewriter rewriter(module.getContext());
  // Analysis and application are consecutive and no IR mutation occurs in
  // between, so rediscover the roots from the module instead of duplicating
  // traversal state in the immutable operation plan.
  for (Operation *root : collectOutermostControlFlowOps(module)) {
    const ControlFlowOpAnalysis *rootAnalysis = plan.lookup(root);
    if (!rootAnalysis) {
      root->emitError("missing frozen control-flow rewrite decision");
      return failure();
    }
    if (!rootAnalysis->needsRewrite())
      continue;
    if (failed(tryDecoupleControlFlowOp(root, rewriter, policy, plan))) {
      root->emitError("failed to apply analyzed pointer decomposition");
      return failure();
    }
  }
  return success();
}

LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy) {
  FailureOr<ControlFlowRewritePlan> plan = analyzeControlFlow(module, policy);
  if (failed(plan))
    return failure();
  return applyControlFlowRewritePlan(module, policy, *plan);
}

} // namespace mlir::triton::controlflow
