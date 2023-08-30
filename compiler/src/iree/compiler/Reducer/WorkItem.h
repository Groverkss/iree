// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_WORK_ITEM_H
#define IREE_COMPILER_REDUCER_WORK_ITEM_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

class WorkItem {
public:
  WorkItem(std::shared_ptr<ModuleOp> root) : root(root), builder(*root) {}

  const ModuleOp &getModule() { return *root; }
  const OpBuilder getBuilder() { return builder; }

  LogicalResult verify() const { return root->verify(); }

  std::unique_ptr<WorkItem> clone() const;
  int64_t getComplexityScore();

private:
  std::shared_ptr<ModuleOp> root;
  OpBuilder builder;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_REDUCER_WORK_ITEM_H
