// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapPatterns.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Visitors.h"

namespace mlir::iree_compiler {

using IREE::VectorExt::VectorLayoutInterface;
using IREE::VectorExt::VectorLayoutOptions;

namespace IREE::Map {

#define GEN_PASS_DEF_GPUMAPVECTORDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Dialect/Map/Transforms/Passes.h.inc"

namespace {

/// VectorLayoutOptions for PackLayout distribution. Returns null for default
/// layouts (0-d vectors are not supported).
class MapVectorLayoutOptions : public VectorLayoutOptions {
public:
  MapVectorLayoutOptions(Operation *root)
      : VectorLayoutOptions(root, /*fullConversion=*/false) {}

  VectorLayoutInterface getDefaultLayout(VectorType type) const override {
    return VectorLayoutInterface();
  }
};

struct GPUMapVectorDistributionPass final
    : impl::GPUMapVectorDistributionPassBase<GPUMapVectorDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Get workgroup size.
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    if (funcOp->hasAttr("workgroup_size")) {
      auto sizes =
          cast<ArrayAttr>(funcOp->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(sizes)) {
        workgroupSize[i] = cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeSize = getWorkgroupSize(funcOp);
      if (!maybeSize) {
        funcOp->emitOpError()
            << "unable to query workgroup_size from entry point";
        return signalPassFailure();
      }
      for (auto [i, v] : llvm::enumerate(*maybeSize)) {
        workgroupSize[i] = v;
      }
    }

    // Create linearized thread ID at function entry.
    IRRewriter rewriter(funcOp);
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    Location loc = funcOp.getLoc();
    SmallVector<Value> threadGrid = {
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::z),
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::y),
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::x)};
    // AffineLinearizeIndexOp expects major-to-minor order (z, y, x).
    std::array<int64_t, 3> reversedWG = {workgroupSize[2], workgroupSize[1],
                                         workgroupSize[0]};
    Value linearThreadId = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, threadGrid, reversedWG, /*disjoint=*/true);

    // Get subgroup size.
    int64_t subgroupSize = getGPUSubgroupSize(funcOp).value_or(32);

    // Populate patterns.
    RewritePatternSet patterns(funcOp.getContext());
    populateGPUDistributionPatterns(patterns);
    populateMapDistributeMemoryOpPatterns(patterns, linearThreadId);
    populateMapDistributeReductionPatterns(patterns, linearThreadId,
                                           subgroupSize);
    populateMapDistributeGenericPatterns(patterns, linearThreadId);
    populateMapDistributeInnerTiledPatterns(patterns);

    MapVectorLayoutOptions options(funcOp);
    if (failed(distributeVectorOps(funcOp, patterns, options))) {
      funcOp->emitOpError() << "failed to distribute vector ops";
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace IREE::Map
} // namespace mlir::iree_compiler
