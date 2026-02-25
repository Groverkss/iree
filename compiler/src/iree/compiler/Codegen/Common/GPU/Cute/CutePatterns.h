// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_CUTE_CUTEPATTERNS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_CUTE_CUTEPATTERNS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Adds PackLayout-specific patterns for distributing vector.transfer_read (and
/// eventually transfer_write) using TVLayoutAttr.
void populateCuteDistributeMemoryOpPatterns(RewritePatternSet &patterns,
                                            Value threadId);

/// Adds PackLayout-specific patterns for distributing vector.multi_reduction
/// using TVLayoutAttr. Handles within-subgroup reductions via local reduce
/// + gpu.subgroup_reduce shuffles.
void populateCuteDistributeReductionPatterns(RewritePatternSet &patterns,
                                             Value threadId,
                                             int64_t subgroupSize);

/// Adds PackLayout-specific patterns for distributing generic vector ops
/// (shape_cast, broadcast, transpose, step) using TVLayoutAttr.
void populateCuteDistributeGenericPatterns(RewritePatternSet &patterns,
                                           Value threadId);

/// Adds PackLayout-specific patterns for distributing iree_codegen.inner_tiled
/// ops.
void populateCuteDistributeInnerTiledPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_CUTE_CUTEPATTERNS_H_
