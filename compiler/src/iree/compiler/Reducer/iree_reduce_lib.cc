// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"

#include "iree/compiler/Reducer/Strategies/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

int mlir::iree_compiler::ireeRunReducingStrategies(Operation *module,
                                                   ReductionStrategy strategy) {

  std::unique_ptr<Pass> strategyPass;
  switch (strategy) {
  case EliminiateDispatches:
    strategyPass = createEliminateDispatchesPass();
    break;
  default:
    llvm_unreachable("Unexpected reduction strategy");
  }

  PassManager pm(module->getContext());
  pm.addPass(std::move(strategyPass));

  if (failed(pm.run(module))) {
    return 1;
  }

  return 0;
}
