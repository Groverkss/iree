// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace IREE::Map {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/Map/Transforms/Passes.h.inc"
} // namespace
} // namespace IREE::Map

void registerIREEMapPasses() {
  // Generated.
  IREE::Map::registerPasses();
}

} // namespace mlir::iree_compiler
