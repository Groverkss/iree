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

ChangeResult DistributionLayout::resolve(AffineMapLayout rhs) {
  AffineMapLayout lhs = vectorLayout;

  // If both layouts are same, do nothing.
  if (lhs == rhs) {
    return ChangeResult::NoChange;
  }

  // Take the more restrictive enforcement.
  if (lhs.state < rhs.state) {
    setState(rhs.state);
    setLayout(rhs.layout);
    return ChangeResult::Change;
  } else if (lhs.state > rhs.state) {
    return ChangeResult::NoChange;
  }

  // From here, both are in the same state, but have different layouts.

  // WeaklyEnforced layouts don't need to be resolved.
  if (lhs.state == Enforcement::WeaklyEnforced) {
    return ChangeResult::NoChange;
  }

  // StronglyEnforced layouts need to be resolved.
  // Layouts have a conflict. Insert a layout resolution operation.
  llvm::errs() << "Layout conflict at: " << *this << "\n";
  llvm::errs() << "With: " << rhs.layout << "\n";
  assert(false && "Layout conflict");
}

ChangeResult DistributionLayout::resolve(const DistributionLayout *rhs) {
  return resolve(rhs->vectorLayout);
}

void DistributionLayout::print(raw_ostream &os) const {
  switch (getState()) {
  case Enforcement::Uninitialized:
    os << "Uninitialized";
    break;
  case Enforcement::WeaklyEnforced:
    os << "WeaklyEnforced";
    break;
  case Enforcement::StronglyEnforced:
    os << "StronglyEnforced";
    break;
    break;
  }

  if (getState() != Enforcement::Uninitialized) {
    os << " " << getLayout();
  }
}

void DistributionLayout::onUpdate(DataFlowSolver *solver) const {
  AnalysisState::onUpdate(solver);

  Value value = point.get<Value>();

  if (propagation) {
    // Make propagation run again on all users of this value.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({user, propagation});
    }
    // TODO: Maybe we need to run it on the parent operation as well to give
    // layout to other results? Seems unlikely though as results usually
    // don't need the same layout?
  }

  if (enforcement) {
    // Make enforcement run on the parent operation.
    if (Operation *definingOp = value.getDefiningOp()) {
      solver->enqueue({definingOp, enforcement});
    }
    // Enforce users of this value also, as some other operands may need to
    // be updated.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({user, enforcement});
    }
  }
}

LogicalResult PropagateLayout::initialize(Operation *op) {
  op->walk([&](Operation *traversed) {
    // If we see a vector.contract, we enforcement it.
    if (auto contractOp = dyn_cast<vector::ContractionOp>(traversed)) {
      Value returnVal = contractOp.getResult();
      DistributionLayout *result = getLatticeElement(returnVal);

      // Get all vector operands of the contract.
      SmallVector<DistributionLayout *> operands;
      operands.reserve(contractOp.getNumOperands());
      for (Value operand : contractOp.getOperands()) {
        if (isa<VectorType>(operand.getType())) {
          operands.push_back(getLatticeElement(operand));
        }
      }

      AffineExpr d0, d1, gpux, gpuy, gpuz;
      bindDims(ctx, d0, d1);
      bindSymbols(ctx, gpux, gpuy, gpuz);

      AffineMap nvidiaMMASyncLayout =
          AffineMap::get(/*newRank=*/2, /*symbolCount=*/3,
                         {
                             gpuy + (8 * d0.floorDiv(2)),
                             gpux + (8 * d1.floorDiv(2)) + (d1 % 2),
                         },
                         ctx);

      // Set result layout.
      result->setState(Enforcement::StronglyEnforced);
      result->setLayout(nvidiaMMASyncLayout);
      propagateIfChanged(result, ChangeResult::Change);

      // Set operand layouts.
      for (DistributionLayout *operand : operands) {
        operand->setState(Enforcement::StronglyEnforced);
        operand->setLayout(nvidiaMMASyncLayout);
        propagateIfChanged(operand, ChangeResult::Change);
      }
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

  // Get the result lattices.
  SmallVector<const DistributionLayout *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    if (!isa<VectorType>(result.getType())) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(result);
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
