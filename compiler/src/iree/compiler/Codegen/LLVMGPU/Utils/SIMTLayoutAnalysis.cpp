// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "SIMTLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTTransferFunctions.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "iree-simt-layout-analysis"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::VectorExt;

DistributionLayout::ResolutionResult
DistributionLayout::doResolution(const AffineMapLayout &rhs) {
  AffineMapLayout &lhs = vectorLayout;

  // If both layouts are same, do nothing.
  if (lhs == rhs) {
    return ResolutionResult::NoChange;
  }

  // Take the other layout if the current layout is empty.
  if (rhs) {
    setInnerLayout(rhs);
    return ResolutionResult::Change;
  }

  // Layouts conflict and need to be resolved.
  return ResolutionResult::Conflict;
}
/// Create an unregistered operation "resolve_conflict" to resolve this
/// conflict.
Operation *createResolutionOp(OpBuilder &builder, Value input) {
  OperationState resolveOpState(input.getLoc(), "resolve_conflict");
  resolveOpState.addOperands(input);
  resolveOpState.addTypes(input.getType());
  Operation *resolution = builder.create(resolveOpState);
  return resolution;
}

ChangeResult
DistributionLayout::resolveWithPossibleConflict(const AffineMapLayout &rhs,
                                                OpOperand &opOperand) {
  ResolutionResult result = doResolution(rhs);

  // If there is no conflict, simply return.
  if (result == ResolutionResult::NoChange) {
    return ChangeResult::NoChange;
  } else if (result == ResolutionResult::Change) {
    return ChangeResult::Change;
  }

  // Resolve conflict by create an operation that takes the input the conflicted
  // value and returns the resolved value.
  OpBuilder builder(opOperand.getOwner());
  Value input = opOperand.get();
  // Create a resolution operation. This conflict should be handeled later by
  // someone else, not this analysis.
  Operation *resolveOp = createResolutionOp(builder, input);
  Value resolvedValue = resolveOp->getResult(0);
  opOperand.set(resolvedValue);

  // Create a new value for the resolved value and subscribe it to propagation
  // and enforcement.
  // We possibly don't need to subscribe this since this value has already
  // reached the top of the lattice and shouldn't do anything else. But it's
  // nicer to do it to have consistency.
  DistributionLayout *resolvedLayout =
      propagation->getLatticeElement(resolvedValue);
  resolvedLayout->subscribeEnforcement(enforcement);

  // We can now resolve this resolved value to the required layout.
  resolvedLayout->resolve(rhs);

  // No change actually needs to be propagated after a conflict resolution.
  // TODO: Ideally, there should be another state in the lattice which says
  // "Fixed", which would say that there is no way you can change this layout
  // anymore, and it should be override any other layout used.
  return ChangeResult::NoChange;
}

ChangeResult
DistributionLayout::resolveWithPossibleConflict(const DistributionLayout *rhs,
                                                OpOperand &opOperand) {
  assert(rhs && "layout to resolve with should not be null");
  return resolveWithPossibleConflict(rhs->vectorLayout, opOperand);
}

ChangeResult DistributionLayout::resolve(const AffineMapLayout &rhs) {
  ResolutionResult result = doResolution(rhs);

  switch (result) {
  case ResolutionResult::NoChange:
    return ChangeResult::NoChange;
  case ResolutionResult::Change:
    return ChangeResult::Change;
  case ResolutionResult::Conflict: {
    llvm::errs() << "Layout conflict at: " << *this << "\n";
    llvm::errs() << "With: " << rhs << "\n";
    llvm_unreachable("Layout conflict should have been handled with "
                     "resolveWithPossibleConflict instead");
  }
  }
}

ChangeResult DistributionLayout::resolve(const DistributionLayout *rhs) {
  assert(rhs && "layout to resolve with should not be null");
  return resolve(rhs->vectorLayout);
}

void DistributionLayout::print(raw_ostream &os) const {
  if (vectorLayout) {
    os << " " << vectorLayout;
  } else {
    os << "Uninitialized";
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
      SmallVector<SmallVector<int64_t>> operandShapes;
      for (Value operand : contractOp.getOperands()) {
        if (isa<VectorType>(operand.getType())) {
          operands.push_back(getLatticeElement(operand));
          SmallVector<int64_t> operandShape{
              cast<VectorType>(operand.getType()).getShape()};
          operandShapes.push_back(operandShape);
        }
      }

#if 1

      auto constructLayout = [&](int64_t canonicalShape, int64_t vectorShape,
                                 ArrayRef<LayoutDimension> dims,
                                 SmallVectorImpl<int64_t> &shapes) {
        int64_t batchDim = vectorShape / canonicalShape;
        shapes.insert(shapes.begin(), batchDim);
        SmallVector<LayoutDimensionAttr> labels;
        auto toAttr = [&](LayoutDimension dim) {
          return LayoutDimensionAttr::get(op->getContext(), dim);
        };
        std::transform(dims.begin(), dims.end(), std::back_inserter(labels),
                       toAttr);
        return PerDimLayoutAttr::get(op->getContext(), labels, shapes);
      };

      SmallVector<int64_t> canonicalShape;
      SmallVector<LayoutDimension> dims;
      SmallVector<int64_t> shapes;

      // --------- A-matrix
      // TODO: Get canonical shapes from instruction shape
      canonicalShape = {16, 16};
      dims = {LayoutDimension::BATCHX, LayoutDimension::LANEY,
              LayoutDimension::VECTORZ};
      shapes = {8, 2};
      PerDimLayoutAttr rowLayout =
          constructLayout(canonicalShape[0], operandShapes[0][0], dims, shapes);

      dims = {LayoutDimension::BATCHY, LayoutDimension::VECTORX,
              LayoutDimension::LANEX, LayoutDimension::VECTORY};
      shapes = {2, 4, 2};
      PerDimLayoutAttr colLayout =
          constructLayout(canonicalShape[1], operandShapes[0][1], dims, shapes);

      SmallVector<PerDimLayoutAttr> layouts;
      layouts.push_back(rowLayout);
      layouts.push_back(colLayout);

      LayoutAttr nvidiaMMASyncLayoutA =
          LayoutAttr::get(op->getContext(), layouts);

      // --------- B-matrix
      canonicalShape = {16, 8};
      dims = {LayoutDimension::BATCHX, LayoutDimension::LANEY,
              LayoutDimension::VECTORZ};
      shapes = {8, 1};
      rowLayout =
          constructLayout(canonicalShape[0], operandShapes[1][0], dims, shapes);

      dims = {LayoutDimension::BATCHY, LayoutDimension::VECTORX,
              LayoutDimension::LANEX, LayoutDimension::VECTORY};
      shapes = {2, 4, 2};
      colLayout =
          constructLayout(canonicalShape[1], operandShapes[0][1], dims, shapes);

      layouts.clear();
      layouts.push_back(rowLayout);
      layouts.push_back(colLayout);

      LayoutAttr nvidiaMMASyncLayoutB =
          LayoutAttr::get(op->getContext(), layouts);

#else
      AffineExpr d0, d1, gpux, gpuy, gpuz;
      bindDims(ctx, d0, d1);
      bindSymbols(ctx, gpux, gpuy, gpuz);

      AffineMap nvidiaMMASyncLayoutA =
          AffineMap::get(/*newRank=*/2, /*symbolCount=*/3,
                         {
                             gpuy + (8 * d0.floorDiv(2)),
                             gpux + (8 * d1.floorDiv(2)) + (d1 % 2),
                         },
                         ctx);

      AffineMap nvidiaMMASyncLayoutB =
          AffineMap::get(/*newRank=*/2, /*symbolCount=*/3,
                         {
                             gpuy,
                             gpux + (8 * d1.floorDiv(2)) + (d1 % 2),
                         },
                         ctx);

      // Get shapes for A, B, C matrix.
      SmallVector<int64_t> aShape = {4, 2};
      SmallVector<int64_t> bShape = {2, 2};
      SmallVector<int64_t> cShape = {2, 2};
#endif

      // Set result layout.
      result->resolve(nvidiaMMASyncLayoutA);
      // Set operand layouts.
      operands[0]->resolve(nvidiaMMASyncLayoutA);
      operands[1]->resolve(nvidiaMMASyncLayoutB);
      operands[2]->resolve(nvidiaMMASyncLayoutA);

      propagateIfChanged(result, ChangeResult::Change);
      propagateIfChanged(operands[0], ChangeResult::Change);
      propagateIfChanged(operands[1], ChangeResult::Change);
      propagateIfChanged(operands[2], ChangeResult::Change);
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

SmallVector<int64_t>
mlir::iree_compiler::getSIMTVectorShape(IREE::VectorExt::LayoutAttr layout) {
  SmallVector<LayoutDimension> dims = {
      LayoutDimension::BATCHX, LayoutDimension::BATCHY,
      LayoutDimension::VECTORZ, LayoutDimension::VECTORY,
      LayoutDimension::VECTORX};
  SmallVector<int64_t> shape = layout.getSIMTVectorShape(dims);
  // TODO: After a projection, the shape could be smaller.
  if (shape.size() < dims.size())
    return shape;
  assert(shape.size() == dims.size());
  // Collapse VectorZ and VectorY dimensions. This is NVIDIA specific.
  shape[2] *= shape[3];
  shape[3] = shape[4];
  shape.pop_back();
  return shape;
}
