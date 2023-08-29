// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/Strategies/Strategies.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static int inputNumber = 0;

static SmallVector<IREE::Util::GlobalOpInterface>
createGlobalRandomInputs(OpBuilder builder, ModuleOp module,
                         ArrayRef<Type> types) {

  builder.setInsertionPointToStart(module.getBody());

  SmallVector<IREE::Util::GlobalOpInterface> globalOps;
  for (auto type : types) {
    auto globalName = "__iree_reduce_" + std::to_string(inputNumber);
    // Create a util.global = util.byte_pattern.
    auto bytePattern =
        builder.getAttr<IREE::Util::BytePatternAttr>(type, inputNumber);
    auto globalOp = builder.create<IREE::Util::GlobalOp>(
        module.getLoc(), globalName, /*isMutable=*/false, type, bytePattern);
    globalOps.push_back(globalOp);
    inputNumber++;
  }

  return globalOps;
}

LogicalResult
mlir::iree_compiler::runEliminateDispatchesStrategy(Operation *root) {
  auto module = dyn_cast<ModuleOp>(root);
  if (!module) {
    return failure();
  }

  OpBuilder builder(module->getContext());

  // Randomly choose a dispatch op.
  // TODO: There is probably a better strategy to choose dispatch ops here.
  std::vector<IREE::Flow::DispatchOp> dispatchOps;
  module->walk([&](IREE::Flow::DispatchOp dispatchOp) {
    dispatchOps.push_back(dispatchOp);
  });

  if (dispatchOps.empty()) {
    return failure();
  }

  int numReplace = 10;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::shuffle(dispatchOps.begin(), dispatchOps.end(), gen);

  // Take the first numReplace dispatch ops.
  ArrayRef<IREE::Flow::DispatchOp> dispatchOpsRef(dispatchOps);
  dispatchOpsRef = dispatchOpsRef.take_front(numReplace);

  for (auto dispatchOp : dispatchOpsRef) {
    // Replace the dispatchOp with random inputs.
    auto resultTypes = dispatchOp.getResults().getTypes();
    std::vector<Type> types(resultTypes.begin(), resultTypes.end());
    auto randomInputs = createGlobalRandomInputs(builder, module, types);

    // Create util.global.load before the dispatchOp.
    SmallVector<Value> newResults;
    builder.setInsertionPoint(dispatchOp);
    for (auto input : randomInputs) {
      auto loadOp =
          builder.create<IREE::Util::GlobalLoadOp>(dispatchOp.getLoc(), input);
      newResults.push_back(loadOp.getResult());
    }

    dispatchOp->replaceAllUsesWith(newResults);
  }

  // Run DCE.
  PassManager pm(module->getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  if (failed(pm.run(module))) {
    return failure();
  }

  return success();
}
