// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/DeltaStratergies.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static IREE::Util::GlobalOpInterface createGlobalRandomInputs(OpBuilder builder,
                                                              Type type) {
  // Generate a random number.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(0, INT64_MAX - 1);
  int64_t inputNumber = dis(gen);
  // Create a util.global = util.byte_pattern.
  auto globalName = "__iree_reduce_" + std::to_string(inputNumber);
  auto bytePattern =
      builder.getAttr<IREE::Util::BytePatternAttr>(type, inputNumber);
  auto globalOp = builder.create<IREE::Util::GlobalOp>(
      builder.getUnknownLoc(), globalName, /*isMutable=*/false, type,
      bytePattern);
  globalOp.setSymVisibility("private");
  return globalOp;
}

static void extractFlowDispatchInModule(ChunkManager &chunker,
                                        WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  SmallVector<IREE::Flow::DispatchOp> dispatchOps;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    for (auto dispatchOp : funcOp.getOps<IREE::Flow::DispatchOp>()) {
      if (!chunker.shouldFeatureBeKept()) {
        dispatchOps.push_back(dispatchOp);
      }
    }
  }

  if (dispatchOps.empty()) {
    return;
  }

  // Replace all dispatch ops with random inputs.
  OpBuilder builder = workItem.getBuilder();
  for (auto dispatchOp : dispatchOps) {
    // Create a random input for each result of the dispatch.
    for (Value result : dispatchOp.getResults()) {
      auto type = result.getType();
      builder.setInsertionPointToStart(module.getBody());
      auto globalOp = createGlobalRandomInputs(builder, type);
      // Create a util.load and replace it with the result.
      builder.setInsertionPoint(dispatchOp);
      auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
          dispatchOp.getLoc(), globalOp);
      result.replaceAllUsesWith(loadOp.getResult());
    }

    // TODO: Use a builder/rewriter here?
    dispatchOp.erase();
  }

  // Simplify.
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  if (failed(pm.run(module))) {
    return;
  }
}

void mlir::iree_compiler::reduceFlowDispatchDelta(Oracle &oracle,
                                                  WorkItem &workItem) {
  runDeltaPass(oracle, workItem, extractFlowDispatchInModule,
               "Reducing Flow Executables");
}
