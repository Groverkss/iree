// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTTransferFunctions.h"

#define DEBUG_TYPE "iree-simt-transfer-functions"

using namespace mlir;
using namespace mlir::iree_compiler;

void iree_compiler::propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Check if one of then operands has an enforced layout and use it.
  for (const DistributionLayout *operandLattice : operandLattices) {
    if (operandLattice->state == Enforcement::Enforced) {
      for (DistributionLayout *resultLattice : resultLattices) {
        ChangeResult changed = resultLattice->resolve(operandLattice);
        LLVM_DEBUG(llvm::dbgs() << "PROPAGATION: resolve prop\n");
        update(resultLattice, changed);
        LLVM_DEBUG(llvm::dbgs() << "PROPAGATION: resolve prop end\n");
      }
      break;
    }
  }
}

void iree_compiler::enforcementTransferFunction(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Check if one of the operands has an enforced layout and use it.
  DistributionLayout *chosenOperandLayout = nullptr;
  for (DistributionLayout *operandLattice : operandLattices) {
    if (operandLattice->state == Enforcement::Enforced) {
      chosenOperandLayout = operandLattice;
      break;
    }
  }

  if (chosenOperandLayout) {
    // Propgate this layout to other operands.
    for (DistributionLayout *operandLattice : operandLattices) {
      ChangeResult changed = operandLattice->resolve(chosenOperandLayout);
      LLVM_DEBUG(llvm::dbgs() << "ENFORCEMENT: resolve prop\n");
      update(operandLattice, changed);
      LLVM_DEBUG(llvm::dbgs() << "ENFORCEMENT: resolve prop end\n");
    }
    return;
  }

  // Check if one of the results has an enforced layout and use it.
  for (const DistributionLayout *resultLattice : resultLattices) {
    if (resultLattice->state == Enforcement::Enforced) {
      for (DistributionLayout *operandLattice : operandLattices) {
        ChangeResult changed = operandLattice->resolve(resultLattice);
        LLVM_DEBUG(llvm::dbgs() << "ENFORCEMENT: resolve prop\n");
        update(operandLattice, changed);
        LLVM_DEBUG(llvm::dbgs() << "ENFORCEMENT: resolve prop end\n");
      }
      return;
    }
  }
}
