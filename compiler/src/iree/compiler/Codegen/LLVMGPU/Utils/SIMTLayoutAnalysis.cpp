// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTTransferFunctions.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "iree-simt-layout-analysis"

using namespace mlir;
using namespace mlir::iree_compiler;

ChangeResult DistributionLayout::resolve(const DistributionLayout *rhs) {
  // If both layouts are same, do nothing.
  if (*this == *rhs) {
    return ChangeResult::NoChange;
  }

  // Take the more restrictive enforcement.
  if (state < rhs->state) {
    state = rhs->state;
    layout = rhs->layout;
    return ChangeResult::Change;
  }

  // Layouts have a conflict. Insert a layout resolution operation.
  llvm::errs() << "Layout conflict: " << *this << " vs " << *rhs << "\n";
  assert(false && "Layout conflict");
}

void DistributionLayout::print(raw_ostream &os) const {
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

void DistributionLayout::onUpdate(DataFlowSolver *solver) const {
  AnalysisState::onUpdate(solver);

  Value value = point.get<Value>();
  LLVM_DEBUG(llvm::dbgs() << "onUpdate: " << value << "\n");

  if (propagation) {
    // Make propagation run again on all users of this value.
    for (Operation *user : value.getUsers()) {
      LLVM_DEBUG(llvm::dbgs() << "Enqueing on PROPAGATION: " << *user << "\n");
      solver->enqueue({user, propagation});
    }
    // TODO: Maybe we need to run it on the parent operation as well to give
    // layout to other results? Seems unlikely though as results usually
    // don't need the same layout?
  }

  if (enforcement) {
    // Make enforcement run on the parent operation.
    if (Operation *definingOp = value.getDefiningOp()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Enqueing on ENFORCEMENT: " << *definingOp << "\n");
      solver->enqueue({definingOp, enforcement});
    }
    // Enforce users of this value also, as some other operands may need to
    // be updated.
    for (Operation *user : value.getUsers()) {
      LLVM_DEBUG(llvm::dbgs() << "Enqueing on ENFORCEMENT: " << *user << "\n");
      solver->enqueue({user, enforcement});
    }
  }
}

LogicalResult PropagateLayout::initialize(Operation *op) {
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

      LLVM_DEBUG(llvm::dbgs() << "contract prop\n");
      propagateIfChanged(layout, ChangeResult::Change);
      LLVM_DEBUG(llvm::dbgs() << "contract prop end\n");
    }

    visitOperation(traversed);
  });

  return success();
}

LogicalResult PropagateLayout::visit(ProgramPoint point) {
  if (Operation *op = dyn_cast_or_null<Operation *>(point)) {
    visitOperation(op);
    return success();
  }

  // Do not expect anything other than an operation.
  return failure();
}

void PropagateLayout::visitOperation(Operation *op) {
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

  LLVM_DEBUG(llvm::dbgs() << "visiting operation: " << *op << "\n");

  for (auto *resultLattice : resultLattices) {
    LLVM_DEBUG(llvm::dbgs() << "result lattice: " << *resultLattice << "\n");
  }

  // Grab the lattice elements of the operands.
  SmallVector<const DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (!isa<VectorType>(operand.getType())) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(operand);
    LLVM_DEBUG(llvm::dbgs() << "operand lattice: " << *operandLattice << "\n");
    operandLattices.push_back(operandLattice);
  }

  auto changeFunc = [&](DistributionLayout *lattice, ChangeResult changed) {
    this->propagateIfChanged(lattice, changed);
  };

  propagationTransferFunction(op, operandLattices, resultLattices, changeFunc);
}

DistributionLayout *PropagateLayout::getLatticeElement(Value val) {
  // Add dependency of operation on the analysis state.
  assert(isa<VectorType>(val.getType()) && "Lattice value should be a vector");
  DistributionLayout *layout =
      DataFlowAnalysis::getOrCreate<DistributionLayout>(val);
  // Subscribe this analysis to updates of the lattice.
  layout->subscribePropagation(this);
  return layout;
}

LogicalResult EnforceLayout::initialize(Operation *op) {
  op->walk([&](Operation *traversed) { visitOperation(traversed); });
  return success();
}

LogicalResult EnforceLayout::visit(ProgramPoint point) {
  if (Operation *op = dyn_cast_or_null<Operation *>(point)) {
    visitOperation(op);
    return success();
  }

  // Do not expect anything other than an operation.
  return failure();
}

void EnforceLayout::visitOperation(Operation *op) {
  // Grab the lattice elements of the operands.
  SmallVector<DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (!isa<VectorType>(operand.getType())) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(operand);
    operandLattices.push_back(operandLattice);
  }

  // Exit early on operations with no results.
  if (operandLattices.size() == 0) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "ENFORCEMENT: visiting operation: " << *op
                          << "\n");

  for (auto *operandLattice : operandLattices) {
    LLVM_DEBUG(llvm::dbgs()
               << "ENFORCEMENT: operand lattice: " << *operandLattice << "\n");
  }

  // Get the result lattices.
  SmallVector<const DistributionLayout *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    if (!isa<VectorType>(result.getType())) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(result);
    LLVM_DEBUG(llvm::dbgs()
               << "ENFORCEMENT: result lattice: " << *resultLattice << "\n");
    resultLattices.push_back(resultLattice);
  }

  auto changeFunc = [&](DistributionLayout *lattice, ChangeResult changed) {
    this->propagateIfChanged(lattice, changed);
  };

  enforcementTransferFunction(op, operandLattices, resultLattices, changeFunc);
}

DistributionLayout *EnforceLayout::getLatticeElement(Value val) {
  // Add dependency of operation on the analysis state.
  assert(isa<VectorType>(val.getType()) && "Lattice value should be a vector");
  DistributionLayout *layout =
      DataFlowAnalysis::getOrCreate<DistributionLayout>(val);
  // Subscribe this analysis to updates of the lattice.
  layout->subscribeEnforcement(this);
  return layout;
}
