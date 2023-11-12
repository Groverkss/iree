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

struct AffineMapLayout {
  bool operator==(const AffineMapLayout &rhs) {
    return state == rhs.state && layout == rhs.layout;
  }

  bool operator!=(const AffineMapLayout &rhs) { return !(*this == rhs); }

  Enforcement state = Enforcement::Uninitialized;
  /// (new vector dims)[gpu syms] -> (old vector dims)
  AffineMap layout;
};

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  Enforcement getState() const { return vectorLayout.state; }
  AffineMap getLayout() const { return vectorLayout.layout; }

  void setState(Enforcement state) { vectorLayout.state = state; }
  void setLayout(AffineMap map) { vectorLayout.layout = map; }

  ChangeResult resolve(const DistributionLayout *rhs);
  ChangeResult resolve(const AffineMapLayout rhs);

  bool isUninitialized() const {
    return getState() == Enforcement::Uninitialized;
  }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return vectorLayout.state == rhs.vectorLayout.state &&
           vectorLayout.layout == rhs.vectorLayout.layout;
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
