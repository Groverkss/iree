// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_PASSES_H_
#define IREE_COMPILER_CODEGEN_PASSES_H_

#include <memory>

namespace mlir {

class Pass;

namespace iree_compiler {

std::unique_ptr<Pass> createEliminateDispatchesPass();

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_PASSES_H_
