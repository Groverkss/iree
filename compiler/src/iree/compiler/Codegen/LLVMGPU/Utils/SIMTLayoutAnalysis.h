// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_

#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

enum class Enforcement {
  Uninitialized = 0,
  WeaklyEnforced = 1,
  StronglyEnforced = 2,
};

class AffineMapLayout {
public:
  AffineMapLayout() = default;

  explicit AffineMapLayout(AffineMap layout, ArrayRef<int64_t> oldShapes)
      : layout(layout), oldShapes(oldShapes) {}

  bool operator==(const AffineMapLayout &rhs) const {
    return layout == rhs.layout && oldShapes == rhs.oldShapes;
  }
  bool operator!=(const AffineMapLayout &rhs) const { return !(*this == rhs); }

  AffineMap getMap() const { return layout; }

  AffineMapLayout permute(ArrayRef<unsigned> permutation) const {
    AffineMap oldLayout = layout;
    AffineMap permuteMap =
        oldLayout.getPermutationMap(permutation, oldLayout.getContext());
    // Permute old vector dims.
    AffineMap newLayout = permuteMap.compose(oldLayout);
    // Permute new vector dims.
    newLayout = newLayout.compose(permuteMap);
    // Permute the shapes.
    SmallVector<int64_t> newOldShapes(oldShapes.size());
    for (unsigned i = 0, e = permutation.size(); i < e; ++i) {
      newOldShapes[i] = oldShapes[permutation[i]];
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
        newOldShapes.push_back(oldShapes[i]);
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
  SmallVector<int64_t> oldShapes;
};

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  Enforcement getState() const { return state; }
  void setState(Enforcement state) { this->state = state; }

  ChangeResult resolve(const DistributionLayout *rhs);
  ChangeResult resolve(Enforcement state, const AffineMapLayout rhs);

  AffineMapLayout getInnerLayout() const { return vectorLayout; }

  bool isUninitialized() const {
    return getState() == Enforcement::Uninitialized;
  }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return state == rhs.state && vectorLayout == rhs.vectorLayout;
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
  void subscribePropagation(DataFlowAnalysis *analysis) {
    propagation = analysis;
  }
  void subscribeEnforcement(DataFlowAnalysis *analysis) {
    enforcement = analysis;
  }

private:
  void setInnerLayout(const AffineMapLayout layout) { vectorLayout = layout; }

  Enforcement state = Enforcement::Uninitialized;
  AffineMapLayout vectorLayout;

  /// A set of analyses that should be updated when this lattice changes.
  DataFlowAnalysis *propagation = nullptr;
  DataFlowAnalysis *enforcement = nullptr;
};

class PropagateLayout : public DataFlowAnalysis {
public:
  explicit PropagateLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override;

  LogicalResult visit(ProgramPoint point) override;

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

private:
  void visitOperation(Operation *op);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
