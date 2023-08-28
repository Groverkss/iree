// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/Strategies/PassDetail.h"
#include "iree/compiler/Reducer/Strategies/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct EliminateDispatchesPass
    : public EliminateDispatchesBase<EliminateDispatchesPass> {
  void runOnOperation() override;

  void getRandomTensorInput();
};

} // namespace

void EliminateDispatchesPass::runOnOperation() {
  Operation *module = getOperation();
  auto funcOps = module->getRegion(0).getOps<func::FuncOp>();

  if (!llvm::hasSingleElement(funcOps)) {
    return signalPassFailure();
  }

  func::FuncOp funcOp = *funcOps.begin();

  // Randomly take a flow.dispatch op and try to replace it.
  auto dispatchOps = funcOp->getRegion(0).getOps<IREE::Flow::DispatchOp>();

  std::vector<IREE::Flow::DispatchOp> dispatchOpsVec(dispatchOps.begin(),
                                                     dispatchOps.end());

  while (!dispatchOpsVec.empty()) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dispatchOpsVec.size() - 1);
    int index = dis(gen);

    IREE::Flow::DispatchOp dispatchOp = dispatchOpsVec[index];
    // Replace all outputs with randomly generated tensor.
    dispatchOp.dump();
    return;
  }
}

std::unique_ptr<Pass> createEliminateDispatchesPass() {
  return std::make_unique<EliminateDispatchesPass>();
}

} // namespace iree_compiler
} // namespace mlir
