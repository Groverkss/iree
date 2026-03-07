// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def GPUMapVectorDistributionPass :
    InterfacePass<"iree-codegen-gpu-map-vector-distribution", "mlir::FunctionOpInterface"> {
  let summary = "Distribute vector ops using PackLayout TV layouts.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect",
  ];
}

#endif // IREE_CODEGEN_DIALECT_MAP_TRANSFORMS_PASSES
