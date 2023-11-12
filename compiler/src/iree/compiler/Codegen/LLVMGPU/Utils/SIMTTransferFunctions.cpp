// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTTransferFunctions.h"

#define DEBUG_TYPE "iree-simt-transfer-functions"

using namespace llvm;
using namespace mlir;
using namespace mlir::iree_compiler;

/// Get a layout if everyone agrees on the same layout.
static DistributionLayout *
getAgreedLayout(ArrayRef<DistributionLayout *> layouts) {
  if (layouts.size() == 0)
    return nullptr;

  // Check if all layouts are same.
  for (unsigned i = 1, e = layouts.size(); i < e; ++i) {
    if (*layouts[i] != *layouts[0]) {
      return nullptr;
    }
  }

  return layouts[0];
}

static const DistributionLayout *
getAgreedLayout(ArrayRef<const DistributionLayout *> layouts) {
  if (layouts.size() == 0)
    return nullptr;

  // Check if all layouts are same.
  for (unsigned i = 1, e = layouts.size(); i < e; ++i) {
    if (*layouts[i] != *layouts[0]) {
      return nullptr;
    }
  }

  return layouts[0];
}

/// Given a list of layouts, agree on a single layout for all of them.
static void enforceSameLayout(
    ArrayRef<DistributionLayout *> layouts,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // TODO: Use the most common layout here.
  // Get the highest enforced layout.
  DistributionLayout *chosenOperandLayout = nullptr;
  for (DistributionLayout *lattice : layouts) {
    if (chosenOperandLayout == nullptr) {
      chosenOperandLayout = lattice;
    } else {
      if (chosenOperandLayout->getState() < lattice->getState()) {
        chosenOperandLayout = lattice;
      }
    }
  }

  if (chosenOperandLayout != nullptr) {
    for (DistributionLayout *lattice : layouts) {
      ChangeResult changed = lattice->resolve(chosenOperandLayout);
      update(lattice, changed);
    }
  }
}

static AffineMap getProjectedLayout(AffineMap oldLayout,
                                    ArrayRef<bool> projectedDims) {
  // Get a new affine map with these dimensions projected out and these results
  // projected out.

  SmallBitVector reductionMaskBV(projectedDims.size());
  SmallVector<unsigned> unreducedPos;
  for (unsigned i = 0, e = projectedDims.size(); i < e; ++i) {
    if (projectedDims[i]) {
      reductionMaskBV = reductionMaskBV.set(i);
    } else {
      unreducedPos.push_back(i);
    }
  }

  AffineMap newLayout =
      projectDims(oldLayout, reductionMaskBV, /*compressDims=*/true);
  newLayout = newLayout.getSubMap(unreducedPos);

  return newLayout;
}

/// =========================
///        PROPAGATION
/// =========================

static void propagateLayoutToElementwiseOp(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  DistributionLayout *result = resultLattices[0];

  // If result lattice already has a strongly enforced layout, we cannot do
  // anything. We do not impose layout conflicts on results.
  if (result->getState() == Enforcement::StronglyEnforced) {
    return;
  }

  // Check if all vector operands agree on the same layout.
  const DistributionLayout *chosenOperandLayout =
      getAgreedLayout(operandLattices);
  if (chosenOperandLayout == nullptr) {
    return;
  }

  ChangeResult changed = result->resolve(chosenOperandLayout);
  update(result, changed);
}

static void propagateLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Multi reduce has only one vector result.
  DistributionLayout *result = resultLattices[0];
  // Multi reduce has second operand as init.
  const DistributionLayout *init = operandLattices[1];

  // If result lattice already has a strongly enforced layout, we cannot do
  // anything. We do not impose layout conflicts on results.
  if (result->getState() == Enforcement::StronglyEnforced) {
    return;
  }

  ChangeResult changed = result->resolve(init);
  update(result, changed);
}

static void propagateLayoutToTransposeOp(
    vector::TransposeOp transpose,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Transpose has only one vector result.
  DistributionLayout *result = resultLattices[0];
  // Transpose has only one vector operand.
  const DistributionLayout *value = operandLattices[0];

  // If result lattice already has a strongly enforced layout, we cannot do
  // anything. We do not impose layout conflicts on results.
  if (result->getState() == Enforcement::StronglyEnforced) {
    return;
  }

  // Cannot propagate layout if value is uninitialized.
  if (value->isUninitialized()) {
    return;
  }

  // Build a transposed layout.
  SmallVector<unsigned> permutation;
  ArrayAttr transp = transpose.getTransp();
  for (Attribute attr : transp) {
    permutation.push_back(attr.cast<IntegerAttr>().getInt());
  }

  AffineMap oldLayout = value->getLayout();
  AffineMap newLayout =
      oldLayout.getPermutationMap(permutation, oldLayout.getContext())
          .compose(oldLayout);

  // Try to resolve with the transposed layout.
  ChangeResult changed = result->resolve({value->getState(), newLayout});
  update(result, changed);
}

void iree_compiler::propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    propagateLayoutToElementwiseOp(op, operandLattices, resultLattices, update);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    propagateLayoutToMultiReductionOp(multiReduce, operandLattices,
                                      resultLattices, update);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    propagateLayoutToTransposeOp(transpose, operandLattices, resultLattices,
                                 update);
    return;
  }

  return;
}

/// =========================
///        ENFORCEMENT
/// =========================

static void enforceLayoutToElementwiseOp(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  // Try to enforce the layout of the result on operands.
  const DistributionLayout *result = resultLattices[0];
  if (!result->isUninitialized()) {
    for (DistributionLayout *operandLattice : operandLattices) {
      ChangeResult changed = operandLattice->resolve(result);
      update(operandLattice, changed);
    }
  }

  // Enforce the same layout on all operands. Note that this will
  // not cause problems with the result because if the result had a strongly
  // enforced layout, it would have enforced its layout on every operand. This
  // is just to handle cases where results have a uninitialized/weakly enforced
  // layout.
  enforceSameLayout(operandLattices, update);
}

static void enforceLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  const DistributionLayout *result = resultLattices[0];
  DistributionLayout *value = operandLattices[0];
  DistributionLayout *init = operandLattices[1];

  // Enforce the result layout on init;
  ChangeResult changedDueToResult = init->resolve(result);

  // Try to make the init agree on the same layout as projected value.
  if (!value->isUninitialized()) {
    SmallVector<bool> reductionMask = multiReduce.getReductionMask();
    AffineMap projectedLayout =
        getProjectedLayout(value->getLayout(), reductionMask);
    ChangeResult changedDueToValue =
        init->resolve({value->getState(), projectedLayout});
    update(init, changedDueToResult | changedDueToValue);
  } else {
    update(init, changedDueToResult);
  }
}

static void enforceLayoutToTransposeOp(
    vector::TransposeOp transpose,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Transpose has only one vector result.
  const DistributionLayout *result = resultLattices[0];
  // Transpose has only one vector operand.
  DistributionLayout *value = operandLattices[0];

  // Cannot enforce layout if result is uninitialized.
  if (result->isUninitialized()) {
    return;
  }

  // Build a transposed layout.
  SmallVector<unsigned> permutation;
  ArrayAttr transp = transpose.getTransp();
  for (Attribute attr : transp) {
    permutation.push_back(attr.cast<IntegerAttr>().getInt());
  }

  AffineMap oldLayout = result->getLayout();
  AffineMap newLayout =
      oldLayout.getPermutationMap(permutation, oldLayout.getContext())
          .compose(oldLayout);

  // Try to resolve with the transposed layout.
  ChangeResult changed = value->resolve({result->getState(), newLayout});
  update(value, changed);
}

void iree_compiler::enforcementTransferFunction(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    enforceLayoutToElementwiseOp(op, operandLattices, resultLattices, update);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    enforceLayoutToMultiReductionOp(multiReduce, operandLattices,
                                    resultLattices, update);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    enforceLayoutToTransposeOp(transpose, operandLattices, resultLattices,
                               update);
    return;
  }
}
