// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H
#define IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H

namespace mlir {

class Operation;

namespace iree_compiler {

int ireeRunReducingStratergies(Operation * module);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H
