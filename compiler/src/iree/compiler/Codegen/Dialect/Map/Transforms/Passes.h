// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Map {
#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Dialect/Map/Transforms/Passes.h.inc" // IWYU pragma: keep
} // namespace mlir::iree_compiler::IREE::Map

namespace mlir::iree_compiler {
/// Register Map dialect transform passes.
void registerIREEMapPasses();
} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES_H_
