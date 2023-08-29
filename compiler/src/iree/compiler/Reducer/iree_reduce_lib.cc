// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"

#include "iree/compiler/Reducer/Strategies/Strategies.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

int mlir::iree_compiler::ireeRunReducingStrategies(Operation *module,
                                                   ReductionStrategy strategy) {

  LogicalResult res = success();

  int curr = 0;

  while (res.succeeded()) {
    switch (strategy) {
    case EliminiateDispatches:
      res = runEliminateDispatchesStrategy(module);
      break;
    default:
      llvm_unreachable("Unexpected reduction strategy");
    }

    if (++curr >= 15) {
      break;
    }
  }

  module->dump();

  return 0;
}
