// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Strategies/PassDetail.h"
#include "iree/compiler/Reducer/Strategies/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct EliminateDispatchesPass
    : public EliminateDispatchesBase<EliminateDispatchesPass> {
  void runOnOperation() override;
};

} // namespace

void EliminateDispatchesPass::runOnOperation() {
  llvm::errs() << "Hello from Eliminate Dispatch Pass\n";
}

std::unique_ptr<Pass> createEliminateDispatchesPass() {
  return std::make_unique<EliminateDispatchesPass>();
}

} // namespace iree_compiler
} // namespace mlir
