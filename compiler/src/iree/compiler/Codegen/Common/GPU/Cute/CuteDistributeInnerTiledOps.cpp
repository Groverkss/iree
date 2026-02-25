// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Cute/CuteDistributionUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Cute/CutePatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

using IREE::Map::PackLayoutAttr;
using VectorValue = TypedValue<VectorType>;

namespace {

/// Collapse distributed (leaf) shape to original-rank shape by merging
/// each original dim's value leaves into one dim.
static Value collapseToOriginalRank(PatternRewriter &rewriter, Location loc,
                                    VectorValue distributed,
                                    PackLayoutAttr layout) {
  SmallVector<int64_t> distShape =
      cast<VectorLayoutInterface>(cast<Attribute>(layout))
          .getDistributedShape();
  SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);
  int32_t rank = layout.getRank();
  SmallVector<int64_t> collapsed(rank, 1);
  for (auto [i, info] : llvm::enumerate(leafMap)) {
    collapsed[info.origDim] *= distShape[i];
  }
  auto ty = VectorType::get(collapsed, distributed.getType().getElementType());
  return vector::ShapeCastOp::create(rewriter, loc, ty, distributed);
}

/// Expand from original-rank shape back to distributed (leaf) shape.
static Value expandToDistributedShape(PatternRewriter &rewriter, Location loc,
                                      Value collapsed, PackLayoutAttr layout) {
  SmallVector<int64_t> distShape =
      cast<VectorLayoutInterface>(cast<Attribute>(layout))
          .getDistributedShape();
  auto ty = VectorType::get(
      distShape, cast<VectorType>(collapsed.getType()).getElementType());
  return vector::ShapeCastOp::create(rewriter, loc, ty, collapsed);
}

struct CuteDistributeInnerTiled final
    : OpDistributionPattern<IREE::Codegen::InnerTiledOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(tiledOp, "requires vector semantics");
    }
    auto semantics =
        dyn_cast<IREE::GPU::InnerTiledSemanticsAttr>(tiledOp.getSemantics());
    if (!semantics) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "expected GPU inner tiled semantics");
    }
    if (semantics.getDistributed()) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "already distributed inner tiled op");
    }

    Location loc = tiledOp.getLoc();
    MLIRContext *ctx = tiledOp.getContext();

    // Distribute and collapse inputs.
    SmallVector<Value> collapsedInputs;
    for (Value input : tiledOp.getInputs()) {
      auto vec = dyn_cast<VectorValue>(input);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp, "expected vector input");
      }
      auto layout = dyn_cast<PackLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp, "missing CuTe layout");
      }
      VectorValue dist = getDistributed(rewriter, vec, signature[vec]);
      collapsedInputs.push_back(
          collapseToOriginalRank(rewriter, loc, dist, layout));
    }

    // Distribute and collapse outputs, collect result layouts.
    SmallVector<Value> collapsedOutputs;
    for (Value output : tiledOp.getOutputs()) {
      auto vec = dyn_cast<VectorValue>(output);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp, "expected vector output");
      }
      auto layout = dyn_cast<PackLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp, "missing CuTe layout");
      }
      VectorValue dist = getDistributed(rewriter, vec, signature[vec]);
      collapsedOutputs.push_back(
          collapseToOriginalRank(rewriter, loc, dist, layout));
    }

    SmallVector<PackLayoutAttr> resultLayouts;
    for (Value result : tiledOp->getResults()) {
      auto vec = dyn_cast<VectorValue>(result);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp, "expected vector result");
      }
      auto layout = dyn_cast<PackLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp, "missing CuTe layout");
      }
      resultLayouts.push_back(layout);
    }

    // Create distributed inner_tiled op.
    auto newSemantics = IREE::GPU::InnerTiledSemanticsAttr::get(
        ctx, /*distributed=*/true, semantics.getOpaque());

    auto newOp = IREE::Codegen::InnerTiledOp::create(
        rewriter, loc, collapsedInputs, collapsedOutputs,
        tiledOp.getIndexingMaps(), tiledOp.getIteratorTypes(),
        tiledOp.getKind(), newSemantics, tiledOp.getPermutations());

    // Expand results back to distributed shape.
    SmallVector<Value> distributedResults;
    for (auto [result, layout] :
         llvm::zip_equal(newOp->getResults(), resultLayouts)) {
      distributedResults.push_back(expandToDistributedShape(
          rewriter, loc, cast<VectorValue>(result), layout));
    }

    replaceOpWithDistributedValues(rewriter, tiledOp, distributedResults);
    return success();
  }
};

} // namespace

void populateCuteDistributeInnerTiledPatterns(RewritePatternSet &patterns) {
  patterns.add<CuteDistributeInnerTiled>(patterns.getContext());
}

} // namespace mlir::iree_compiler
