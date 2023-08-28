// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_REDUCER_STRATEGIES_PASSES
#define IREE_REDUCER_STRATEGIES_PASSES

include "mlir/Pass/PassBase.td"

def EliminateDispatches
    : Pass<"eliminate-dispatches"> {
  let summary = "Eliminates Dispatches";
  let constructor = "mlir::iree_compiler::createEliminateDispatchesPass()";
}

#endif // IREE_REDUCER_STRATEGIES_PASSES
