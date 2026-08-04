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

#ifndef TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
#define TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H

#include "TritonControlFlowOpt/ControlFlowAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::controlflow {

/// Handoff marker for SCF loops whose pointer slots have already been expanded
/// into policy-owned descriptor components. TritonToLinalg consumes and
/// removes it instead of applying its legacy pointer-loop decomposition a
/// second time.
inline constexpr llvm::StringLiteral kPointerDescriptorBoundaryAttr =
    "PointerDescriptorBoundary";

/// Policy-owned description of one value crossing a control-flow boundary.
///
/// `components` contain every runtime value needed to rebuild the original
/// value. A policy may place a selected subset in an expanded SCF signature.
/// `attributes` retain non-SSA metadata. Both layouts are private to the
/// policy; the shared rewrite never interprets pointer-specific fields.
struct DecomposedValue {
  Type originalType;
  SmallVector<Value> components;
  SmallVector<Attribute> attributes;
};

/// Read-only view of the SSA mapping and decompositions visible at the current
/// rewrite point. Policies use it while recursively analyzing pointer
/// producers; the state exists only for one control-flow rewrite attempt.
class ControlFlowRewriteContext {
public:
  ControlFlowRewriteContext(
      const IRMapping &valueMapping,
      const llvm::DenseMap<Value, DecomposedValue> &decomposedValues)
      : valueMapping(valueMapping), decomposedValues(decomposedValues) {}

  Value remap(Value value) const;
  const DecomposedValue *lookup(Value value) const;

private:
  const IRMapping &valueMapping;
  const llvm::DenseMap<Value, DecomposedValue> &decomposedValues;
};

/// Pointer-semantics interface implemented by each decomposition policy.
///
/// The policy decides how its value is decomposed and rebuilt, which components
/// cross loop/if boundaries, and whether two decompositions share a compatible
/// non-carried schema. It carries no mutable state between policy invocations;
/// the optional finalization hook may attach a downstream handoff marker.
class ControlFlowRewritePolicy : public ControlFlowAnalysisPolicy {
public:
  virtual ~ControlFlowRewritePolicy() = default;

  /// Whether results of this ordinary operation need immediate decomposition
  /// after cloning so later operations can reuse their exact component state.
  virtual bool shouldDecomposeOperation(Operation *op) const = 0;

  /// Allows a policy to annotate a newly created SCF operation with metadata
  /// required by a downstream consumer. The shared rewrite invokes this after
  /// copying the original attributes and only when this policy changed the
  /// operation's own signature, so metadata cannot leak to an outer operation
  /// that was rebuilt solely because it contains nested control flow.
  virtual void finalizeRewrittenControlFlowOp(Operation *op) const {}

  virtual FailureOr<DecomposedValue>
  decompose(Value value, const ControlFlowRewriteContext &context,
            OpBuilder &builder, Location loc) const = 0;

  virtual Value recompose(const DecomposedValue &value, OpBuilder &builder,
                          Location loc) const = 0;
};

/// Rewrites supported SCF operations from outermost to innermost.
///
/// Applies a previously frozen plan without running value analysis again.
/// Signature expansion, region cloning, terminator rewriting, nested recursion
/// and result replacement are driven solely by operation-level decisions in
/// `plan`.
LogicalResult
applyControlFlowRewritePlan(ModuleOp module,
                            const ControlFlowRewritePolicy &policy,
                            const ControlFlowRewritePlan &plan);

/// Analyzes the complete decomposition stage before mutating the IR, then
/// applies the frozen plan from outermost to innermost. Pointer semantics
/// remain selected by `policy` so different decompositions share the same SCF
/// plumbing.
LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy);

} // namespace mlir::triton::controlflow

#endif // TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
