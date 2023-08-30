// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"

#include "iree/compiler/Reducer/DeltaStratergies.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace llvm;

int mlir::iree_compiler::ireeRunReducingStrategies(Operation *module,
                                                   StringRef testScript) {
  auto root = std::make_shared<ModuleOp>(module);
  std::unique_ptr<WorkItem> workItem = std::make_unique<WorkItem>(root);
  Oracle oracle(testScript);
  reduceFlowDispatchDelta(oracle, *workItem);
  return 0;
}
