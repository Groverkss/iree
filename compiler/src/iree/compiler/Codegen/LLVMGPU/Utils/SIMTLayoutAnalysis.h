// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

enum class Enforcement {
  Uninitialized = 0,
  NeedsEnforcement = 1,
  Enforced = 2
};

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  ChangeResult resolve(const DistributionLayout *rhs) {
    // If both layouts are same, do nothing.
    if (*this == *rhs) {
      return ChangeResult::NoChange;
    }

    // Take the more restrictive enforcement.
    if (state < rhs->state) {
      *this = *rhs;
      return ChangeResult::Change;
    }

    // Layouts have a conflict. Insert a layout resolution operation.
    llvm::errs() << "Layout conflict: " << *this << " vs " << *rhs << "\n";
    assert(false && "Layout conflict");
  }

  bool isUninitialized() const { return state == Enforcement::Uninitialized; }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return state == rhs.state && layout == rhs.layout;
  }

  void print(raw_ostream &os) const override {
    if (state == Enforcement::Uninitialized) {
      os << "Uninitialized";
      return;
    }

    if (state == Enforcement::NeedsEnforcement) {
      os << "NeedsEnforcement";
      return;
    }

    if (state == Enforcement::Enforced) {
      os << "Enforced: "
         << " -> " << layout;
      return;
    }
  }

  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override {
    AnalysisState::onUpdate(solver);

    Value value = point.get<Value>();

    if (propagation) {
      // Make propagation run again on all users of this value.
      for (Operation *user : value.getUsers())
        solver->enqueue({user, propagation});
      // TODO: Maybe we need to run it on the parent operation as well to give
      // layout to other results? Seems unlikely though as results usually don't
      // need the same layout?
    }

    if (enforcement) {
      // Make enforcement run on the parent operation.
      if (Operation *definingOp = value.getDefiningOp())
        solver->enqueue({definingOp, enforcement});
      // Enforce users of this value also, as some other operands may need to be
      // updated.
      for (Operation *user : value.getUsers())
        solver->enqueue({user, enforcement});
    }
  }

  /// Subscribe an analysis to updates of the lattice. When the lattice changes,
  /// subscribed analyses are re-invoked. This is
  /// more efficient than relying on the dependency map.
  void subscribePropagation(DataFlowAnalysis *analysis) {
    propagation = analysis;
  }
  void subscribeEnforcement(DataFlowAnalysis *analysis) {
    enforcement = analysis;
  }

  Enforcement state = Enforcement::Uninitialized;
  /// (new vector dims)[gpu syms] -> (old vector dims)
  AffineMap layout;

private:
  /// A set of analyses that should be updated when this lattice changes.
  DataFlowAnalysis *propagation = nullptr;
  DataFlowAnalysis *enforcement = nullptr;
};

class PropagateLayout : public DataFlowAnalysis {
public:
  explicit PropagateLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override {
    op->walk([&](Operation *traversed) {
      // If we see a vector.contract, we enforcement it.
      if (auto contractOp = dyn_cast<vector::ContractionOp>(traversed)) {
        Value returnVal = contractOp.getResult();
        DistributionLayout *layout = getLatticeElement(returnVal);
        layout->state = Enforcement::Enforced;

        AffineExpr d0;
        bindDims(ctx, d0);
        layout->layout = AffineMap::get(/*newRank=*/2,
                                        /*gpuSums=*/3, {d0, d0}, ctx);

        propagateIfChanged(layout, ChangeResult::Change);
      }

      visitOperation(traversed);
    });

    return success();
  }

  LogicalResult visit(ProgramPoint point) override {
    if (Operation *op = dyn_cast_or_null<Operation *>(point)) {
      visitOperation(op);
      return success();
    }

    // Do not expect anything other than an operation.
    return failure();
  }

private:
  void visitOperation(Operation *op) {
    // Get the result lattices.
    SmallVector<DistributionLayout *> resultLattices;
    resultLattices.reserve(op->getNumResults());
    for (Value result : op->getResults()) {
      if (!isa<VectorType>(result.getType())) {
        continue;
      }

      DistributionLayout *resultLattice = getLatticeElement(result);
      resultLattices.push_back(resultLattice);
    }

    // Exit early on operations with no results.
    if (resultLattices.size() == 0) {
      return;
    }

    // Grab the lattice elements of the operands.
    SmallVector<const DistributionLayout *> operandLattices;
    operandLattices.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      if (!isa<VectorType>(operand.getType())) {
        continue;
      }

      DistributionLayout *operandLattice = getLatticeElement(operand);
      operandLattices.push_back(operandLattice);
    }

    visitOperationImpl(op, operandLattices, resultLattices);
  }

  void visitOperationImpl(Operation *op,
                          ArrayRef<const DistributionLayout *> operandLattices,
                          ArrayRef<DistributionLayout *> resultLattices) {
    // Check if one of the operands has an enforced layout and use it.
    for (const DistributionLayout *operandLattice : operandLattices) {
      if (operandLattice->state == Enforcement::Enforced) {
        for (DistributionLayout *resultLattice : resultLattices) {
          ChangeResult changed = resultLattice->resolve(operandLattice);
          propagateIfChanged(resultLattice, changed);
        }
        return;
      }
    }
  }

  DistributionLayout *getLatticeElement(Value val) {
    // Add dependency of operation on the analysis state.
    assert(isa<VectorType>(val.getType()) &&
           "Lattice value should be a vector");
    DistributionLayout *layout =
        DataFlowAnalysis::getOrCreate<DistributionLayout>(val);
    // Subscribe this analysis to updates of the lattice.
    layout->subscribePropagation(this);
    return layout;
  }

  MLIRContext *ctx;
};

}; // namespace iree_compiler
}; // namespace mlir
