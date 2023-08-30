// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/WorkItem.h"

using namespace mlir;
using namespace mlir::iree_compiler;

std::unique_ptr<WorkItem> WorkItem::clone() const {
  return std::make_unique<WorkItem>(std::make_shared<ModuleOp>(root->clone()));
}

int64_t WorkItem::getComplexityScore() {
  // TODO: Guide the reducer using this complexity score.
  return 0;
}
