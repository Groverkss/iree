// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

/// Forward decleration for analysis.
class PropagateLayout;
class EnforceLayout;

#if 1
using AffineMapLayout = IREE::VectorExt::LayoutAttr;
SmallVector<int64_t> getSIMTVectorShape(IREE::VectorExt::LayoutAttr layout);
#else
class AffineMapLayout {
public:
  AffineMapLayout() = default;

  explicit AffineMapLayout(AffineMap layout, ArrayRef<int64_t> simtShape)
      : layout(layout), simtShape(simtShape) {}

  bool operator==(const AffineMapLayout &rhs) const {
    return layout == rhs.layout && simtShape == rhs.simtShape;
  }
  bool operator!=(const AffineMapLayout &rhs) const { return !(*this == rhs); }

  AffineMap getMap() const { return layout; }
  ArrayRef<int64_t> getSimtShape() const { return simtShape; }

  AffineMapLayout permute(ArrayRef<unsigned> permutation) const {
    AffineMap oldLayout = layout;
    AffineMap permuteMap =
        oldLayout.getPermutationMap(permutation, oldLayout.getContext());
    // Permute old vector dims.
    AffineMap newLayout = permuteMap.compose(oldLayout);
    // Permute new vector dims.
    newLayout = newLayout.compose(permuteMap);
    // Permute the shapes.
    SmallVector<int64_t> newOldShapes(simtShape.size());
    for (unsigned i = 0, e = permutation.size(); i < e; ++i) {
      newOldShapes[i] = simtShape[permutation[i]];
    }
    return AffineMapLayout(newLayout, newOldShapes);
  }

  AffineMapLayout project(ArrayRef<bool> projectedDims) {
    // Get a new affine map with these dimensions projected out and these
    // results projected out.

    llvm::SmallBitVector reductionMaskBV(projectedDims.size());
    SmallVector<unsigned> unreducedPos;
    for (unsigned i = 0, e = projectedDims.size(); i < e; ++i) {
      if (projectedDims[i]) {
        reductionMaskBV = reductionMaskBV.set(i);
      } else {
        unreducedPos.push_back(i);
      }
    }

    AffineMap newLayout =
        projectDims(layout, reductionMaskBV, /*compressDims=*/true);
    newLayout = newLayout.getSubMap(unreducedPos);

    // Project the shapes.
    SmallVector<int64_t> newOldShapes;
    for (unsigned i = 0, e = projectedDims.size(); i < e; ++i) {
      if (!projectedDims[i]) {
        newOldShapes.push_back(simtShape[i]);
      }
    }
    return AffineMapLayout(newLayout, newOldShapes);
  }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const AffineMapLayout &layout) {
    layout.print(os);
    return os;
  }

  void print(raw_ostream &os) const { layout.print(os); }

private:
  /// (new vector dims)[gpu syms] -> (old vector dims)
  AffineMap layout;
  SmallVector<int64_t> simtShape;
};
#endif

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  Value getValue() const {
    ProgramPoint point = getPoint();
    assert(isa<Value>(point) && "expected program point to be a value");
    return cast<Value>(point);
  }

  ChangeResult resolveWithPossibleConflict(const DistributionLayout *rhs,
                                           OpOperand &operand);
  ChangeResult resolveWithPossibleConflict(const AffineMapLayout &rhs,
                                           OpOperand &operand);

  ChangeResult resolve(const DistributionLayout *rhs);
  ChangeResult resolve(const AffineMapLayout &rhs);

  AffineMapLayout getInnerLayout() const { return vectorLayout; }

  bool isUninitialized() const { return !vectorLayout; }
  bool hasLayout() const { return !isUninitialized(); }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return vectorLayout == rhs.vectorLayout;
  }
  bool operator!=(const DistributionLayout &rhs) const {
    return !(*this == rhs);
  }

  void print(raw_ostream &os) const override;

  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice
  /// changes, subscribed analyses are re-invoked. This is more efficient than
  /// relying on the dependency map.
  void subscribePropagation(PropagateLayout *analysis) {
    propagation = analysis;
  }
  void subscribeEnforcement(EnforceLayout *analysis) { enforcement = analysis; }

private:
  /// The result of a resolution.
  /// Change: The layout was changed.
  /// Conflict: The layout was not changed because there was a conflict.
  /// NoChange: The layout was not changed because it was already the same.
  enum ResolutionResult {
    Change,
    Conflict,
    NoChange,
  };

  /// Attempt to resolve the current lattice with the given lattice. Returns if
  /// the current layout was not changed, changed or if there was a layout
  /// conflict.
  ResolutionResult doResolution(const AffineMapLayout &rhs);

  /// Set the layout for this lattice element to the given layout. This function
  /// should only be used when you know there will be no layout conflicts.
  /// Otherwise, the resolve-like functions should be used.
  void setInnerLayout(const AffineMapLayout &layout) { vectorLayout = layout; }

  /// The layout of the vector SSA Value.
  AffineMapLayout vectorLayout;

  /// Each lattice element stores a pointer to the analysis that work on it so
  /// it can notify them when it changes.
  PropagateLayout *propagation = nullptr;
  EnforceLayout *enforcement = nullptr;
};

class PropagateLayout : public DataFlowAnalysis {
public:
  explicit PropagateLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override;

  LogicalResult visit(ProgramPoint point) override;

  /// Register a new value to be part of the dataflow analysis. The value should
  /// not be part of the analysis already. This is used for new values that are
  /// created.
  void registerNewValue(Value val, const AffineMapLayout &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

class EnforceLayout : public DataFlowAnalysis {
public:
  explicit EnforceLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override;

  LogicalResult visit(ProgramPoint point) override;

  void registerNewValue(Value val, const AffineMapLayout &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
