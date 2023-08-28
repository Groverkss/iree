// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_STRATEGIES_PASS_DETAIL_H_
#define IREE_COMPILER_REDUCER_STRATEGIES_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

#define GEN_PASS_CLASSES
#include "iree/compiler/Reducer/Strategies/Passes.h.inc"

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_REDUCER_STRATEGIES_PASS_DETAIL_H_
