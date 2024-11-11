// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_GPUMARKERS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_GPUMARKERS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

namespace mlir::iree_compiler::gpu_markers {

void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, ArrayRef<int64_t> basis,
              ArrayRef<int64_t> mapping);

LogicalResult getBasis(IREE::GPU::LoweringConfigAttr config,
                       IREE::GPU::TilingLevel level,
                       SmallVector<int64_t> &basis,
                       SmallVector<int64_t> &mapping);

} // namespace mlir::iree_compiler::gpu_markers

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUMARKERS_H_
