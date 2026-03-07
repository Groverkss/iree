// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapPatterns.h"

#include <numeric>

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapDistributionUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

using IREE::Map::filterZeros;
using IREE::Map::getLeafValue;
using IREE::Map::getLeaves;
using IREE::Map::isLeaf;
using IREE::Map::makeLeaf;
using IREE::Map::makeTuple;
using IREE::Map::PackLayoutAttr;
using IREE::Map::PackMapAttr;
using IREE::VectorExt::DistributionSignature;
using IREE::VectorExt::OpDistributionPattern;
using IREE::VectorExt::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

namespace {

//===----------------------------------------------------------------------===//
// MapDistributeMultiReduction
//===----------------------------------------------------------------------===//

/// Build a layout mapping virtual reduction-thread positions -> actual thread
/// IDs. Projects out parallel dims, flattens, filters stride-0/size-1
/// leaves, sorts by stride descending, and coalesces.
///
/// Because the reduction is associative, we can freely reorder the thread
/// contributions. Sorting by stride groups nearby thread IDs together, and
/// coalescing merges contiguous modes -- even across different original dims.
/// Parallel dims may create stride gaps that prevent full coalescing.
static PackMapAttr buildReductionThreadGrid(MLIRContext *ctx,
                                            PackLayoutAttr layout,
                                            ArrayRef<bool> reductionMask) {
  // 1. Project out parallel dims -> keep only reduction dims.
  SmallVector<bool> dropParallel(reductionMask.size());
  for (size_t i = 0; i < reductionMask.size(); ++i) {
    dropParallel[i] = !reductionMask[i];
  }
  PackMapAttr redOnly = layout.getMap().project(dropParallel);

  // 2. Flatten + filter zeros -> flat list of thread-only leaves.
  PackMapAttr flat = redOnly.flatten();
  auto [filtS, filtD] = filterZeros(ctx, flat.getShape(), flat.getStride());

  // 3. Sort by stride descending (lex: smallest stride rightmost).
  SmallVector<int32_t> shapes = getLeaves(filtS);
  SmallVector<int32_t> strides = getLeaves(filtD);
  SmallVector<unsigned> indices(shapes.size());
  std::iota(indices.begin(), indices.end(), 0);
  llvm::sort(indices,
             [&](unsigned a, unsigned b) { return strides[a] > strides[b]; });

  SmallVector<Attribute> sortedS, sortedD;
  for (unsigned i : indices) {
    sortedS.push_back(makeLeaf(ctx, shapes[i]));
    sortedD.push_back(makeLeaf(ctx, strides[i]));
  }

  // 4. Build layout and coalesce.
  return PackMapAttr::get(ctx, makeTuple(ctx, sortedS), makeTuple(ctx, sortedD))
      .coalesce();
}

/// Compute how many contiguous flat positions in the reduction grid fit
/// within one subgroup.
///
/// Requires: `grid` is coalesced (flat, sorted by stride descending --
/// as returned by buildReductionThreadGrid).
///
/// The grid is in lex order (smallest stride rightmost = fastest-varying).
/// Walking right-to-left, each mode's positions advance the thread ID by
/// `stride` per step. We accumulate the max thread ID and stop when adding
/// another position would exceed sgSize.
///
/// Example: grid = (2, 16) : (64, 1), sgSize = 32.
///   Mode (16):(1): fit = min(16, 31/1+1) = 16, maxTid = 15.
///   Mode (2):(64): fit = min(2, (31-15)/64+1) = 1, maxTid = 15. Break.
///   K = 16. The stride-64 gap means only 16 of 32 grid positions fit in
///   one subgroup. logicalDivide(grid, (16):(1)) splits within/across SG.
static int64_t computeSubgroupTilerSize(PackMapAttr grid, int64_t sgSize) {
  // Grid is coalesced: iterate modes directly (no flattening needed).
  int32_t n = grid.getRank();
  int64_t K = 1;
  int64_t maxTid = 0;
  // Walk right-to-left (smallest stride first in lex order).
  for (int32_t i = n - 1; i >= 0; --i) {
    int64_t s = getLeafValue(grid.getShapeMode(i));
    int64_t d = getLeafValue(grid.getStrideMode(i));
    if (d == 0) {
      K *= s;
      continue;
    }
    int64_t fit = std::min(s, (sgSize - 1 - maxTid) / d + 1);
    K *= fit;
    maxTid += (fit - 1) * d;
    if (fit < s) {
      break;
    }
  }
  return K;
}

struct MapDistributeMultiReduction final
    : OpDistributionPattern<vector::MultiDimReductionOp> {
  MapDistributeMultiReduction(MLIRContext *ctx, Value threadId,
                              int64_t subgroupSize)
      : OpDistributionPattern(ctx), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reduceOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto srcVector = cast<VectorValue>(reduceOp.getSource());
    auto layout = dyn_cast<PackLayoutAttr>(signature[srcVector]);
    if (!layout) {
      return rewriter.notifyMatchFailure(reduceOp, "not a PackLayout");
    }

    int64_t rank = layout.getRank();
    SmallVector<bool> reductionMask = reduceOp.getReductionMask();
    vector::CombiningKind kind = reduceOp.getKind();
    MLIRContext *ctx = rewriter.getContext();
    Location loc = reduceOp.getLoc();

    // =================================================================
    // Step 1: Local reduce -- reduce per-thread value leaves.
    // =================================================================
    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);

    // Map distributed dims to original dims via leaf map.
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);
    SmallVector<int64_t> distributedReductionDims;
    for (auto [i, info] : llvm::enumerate(leafMap)) {
      if (reductionMask[info.origDim]) {
        distributedReductionDims.push_back(i);
      }
    }

    // Compute the locally-reduced type.
    SmallVector<int64_t> disShape(disSrc.getType().getShape());
    Type elemTy = disSrc.getType().getElementType();

    SmallVector<int64_t> localReducedShape;
    for (auto [i, s] : llvm::enumerate(disShape)) {
      if (!reductionMask[leafMap[i].origDim]) {
        localReducedShape.push_back(s);
      }
    }

    Type localReducedTy;
    if (localReducedShape.empty()) {
      localReducedTy = elemTy;
    } else {
      localReducedTy = VectorType::get(localReducedShape, elemTy);
    }

    Value localInit =
        getCombiningIdentityValue(loc, rewriter, kind, localReducedTy);

    Value localReduced = vector::MultiDimReductionOp::create(
        rewriter, loc, localReducedTy, kind, disSrc, localInit,
        distributedReductionDims);

    // =================================================================
    // Step 2: Build the reduction thread grid.
    //
    // Project out parallel dims, flatten, filter stride-0/size-1, sort by
    // stride descending (lex convention), and coalesce. Sorting is valid
    // because the reduction is associative.
    // =================================================================
    PackMapAttr grid = buildReductionThreadGrid(ctx, layout, reductionMask);
    int64_t gridSize = grid.getSize();

    // =================================================================
    // Step 3: Split the grid into within-subgroup and across-subgroup
    //   parts using logicalDivide.
    // =================================================================
    int64_t K = computeSubgroupTilerSize(grid, subgroupSize);
    bool hasCrossSubgroup = (K < gridSize);

    PackMapAttr withinSGMap = grid;
    PackMapAttr acrossSGMap;

    if (hasCrossSubgroup) {
      PackMapAttr tiler = PackMapAttr::makeIdentity(ctx, {K});
      PackMapAttr divided = grid.logicalDivide(tiler);
      // Mode 0 = tiler part = within-subgroup threads.
      // Mode 1 = complement = across-subgroup factor.
      Attribute withinShape = divided.getShapeMode(0);
      Attribute withinStride = divided.getStrideMode(0);
      withinSGMap = PackMapAttr::get(
          ctx,
          isLeaf(withinShape) ? makeTuple(ctx, {withinShape}) : withinShape,
          isLeaf(withinStride) ? makeTuple(ctx, {withinStride}) : withinStride);

      Attribute acrossShape = divided.getShapeMode(1);
      Attribute acrossStride = divided.getStrideMode(1);
      acrossSGMap = PackMapAttr::get(
          ctx,
          isLeaf(acrossShape) ? makeTuple(ctx, {acrossShape}) : acrossShape,
          isLeaf(acrossStride) ? makeTuple(ctx, {acrossStride}) : acrossStride);
    }

    // =================================================================
    // Step 4: Thread shuffle (within-subgroup).
    // =================================================================
    struct ClusterInfo {
      int64_t size;
      int64_t stride;
    };
    SmallVector<ClusterInfo> clusters;
    if (gridSize > 1) {
      PackMapAttr withinCoalesced = withinSGMap.coalesce();
      SmallVector<int32_t> clusterShapes =
          getLeaves(withinCoalesced.getShape());
      SmallVector<int32_t> clusterStrides =
          getLeaves(withinCoalesced.getStride());
      for (auto [s, d] : llvm::zip(clusterShapes, clusterStrides)) {
        if (s > 1 && d > 0) {
          clusters.push_back({s, d});
        }
      }
    }

    Value shuffled = localReduced;
    if (!clusters.empty()) {
      // Flatten to 1D for element-wise shuffle.
      auto localReducedVecTy = dyn_cast<VectorType>(localReduced.getType());
      int64_t numElements =
          localReducedVecTy ? localReducedVecTy.getNumElements() : 1;

      // For scalar results, wrap in vector<1x> for uniform handling.
      bool isScalar = !localReducedVecTy;
      if (isScalar) {
        auto vec1Ty = VectorType::get({1}, elemTy);
        localReduced =
            vector::BroadcastOp::create(rewriter, loc, vec1Ty, localReduced);
        numElements = 1;
      }

      auto flatTy = VectorType::get({numElements}, elemTy);
      Value flat =
          vector::ShapeCastOp::create(rewriter, loc, flatTy, localReduced);

      for (int64_t i = 0; i < numElements; ++i) {
        Value elem = vector::ExtractOp::create(rewriter, loc, flat, i);
        for (auto [clusterSize, clusterStride] : clusters) {
          elem = gpu::SubgroupReduceOp::create(
              rewriter, loc, elem, combiningKindToAllReduce(kind),
              /*uniform=*/false, /*cluster_size=*/clusterSize,
              /*cluster_stride=*/clusterStride);
        }
        flat = vector::InsertOp::create(rewriter, loc, elem, flat, i);
      }

      if (isScalar) {
        shuffled = vector::ExtractOp::create(rewriter, loc, flat, 0);
      } else {
        shuffled =
            vector::ShapeCastOp::create(rewriter, loc, localReducedVecTy, flat);
      }
    }

    if (!hasCrossSubgroup) {
      // Combine with accumulator and replace.
      Value acc = reduceOp.getAcc();
      auto accVector = dyn_cast<VectorValue>(acc);
      Value disAcc =
          accVector ? getDistributed(rewriter, accVector, signature[accVector])
                    : acc;

      Value result =
          vector::makeArithReduction(rewriter, loc, kind, shuffled, disAcc);

      if (auto resVector = dyn_cast<VectorValue>(reduceOp.getResult())) {
        replaceOpWithDistributedValues(rewriter, reduceOp, result);
      } else {
        rewriter.replaceOp(reduceOp, result);
      }
      return success();
    }

    // =================================================================
    // Step 5: Cross-subgroup reduction via shared memory.
    // =================================================================

    // Build write/read layouts: parallel modes + appended across-SG mode.
    SmallVector<Attribute> writeShapes, writeStrides;
    SmallVector<Attribute> readShapes, readStrides;
    for (int32_t d = 0; d < rank; ++d) {
      if (!reductionMask[d]) {
        writeShapes.push_back(layout.getShapeMode(d));
        writeStrides.push_back(layout.getStrideMode(d));
        readShapes.push_back(layout.getShapeMode(d));
        readStrides.push_back(layout.getStrideMode(d));
      }
    }
    // Append across-SG modes: write with actual strides, read with stride 0.
    // Coalesce first so all modes are leaves -- otherwise a hierarchical shape
    // mode paired with a scalar stride 0 would violate congruence.
    PackMapAttr acrossCoalesced = acrossSGMap.coalesce();
    for (int32_t i = 0; i < acrossCoalesced.getRank(); ++i) {
      writeShapes.push_back(acrossCoalesced.getShapeMode(i));
      writeStrides.push_back(acrossCoalesced.getStrideMode(i));
      readShapes.push_back(acrossCoalesced.getShapeMode(i));
      readStrides.push_back(makeLeaf(ctx, 0));
    }

    PackLayoutAttr writeLayout = PackLayoutAttr::get(
        ctx, makeTuple(ctx, writeShapes), makeTuple(ctx, writeStrides));
    PackLayoutAttr readLayout = PackLayoutAttr::get(
        ctx, makeTuple(ctx, readShapes), makeTuple(ctx, readStrides));
    VectorLayoutInterface writeLayoutIface(writeLayout);
    VectorLayoutInterface readLayoutIface(readLayout);

    // Reshape shuffled result to write layout's distributed shape.
    SmallVector<int64_t> writeDistShape =
        writeLayoutIface.getDistributedShape();
    auto writeDistTy = VectorType::get(writeDistShape, elemTy);
    Value isoRank;
    if (isa<VectorType>(shuffled.getType())) {
      isoRank =
          vector::ShapeCastOp::create(rewriter, loc, writeDistTy, shuffled);
    } else {
      isoRank =
          vector::BroadcastOp::create(rewriter, loc, writeDistTy, shuffled);
    }

    // ToSIMDOp: distributed -> undistributed (= buffer shape).
    auto bufferVecTy =
        VectorType::get(writeLayoutIface.getUndistributedShape(), elemTy);
    VectorValue toSIMD =
        IREE::VectorExt::ToSIMDOp::create(rewriter, loc, bufferVecTy, isoRank);

    // Emit to_layout to redistribute from writeLayout to readLayout.
    // MapDistributeToLayout handles the shared memory
    // alloc/barrier/write/barrier/read.
    auto toLayout = IREE::VectorExt::ToLayoutOp::create(
        rewriter, loc, toSIMD, VectorLayoutInterface(readLayout));
    setSignatureForRedistribution(rewriter, toLayout, {writeLayoutIface},
                                  {readLayoutIface});

    // Second multi_reduction over the appended SG dims.
    int64_t writeRank = writeLayout.getRank();
    int64_t numParallelDims = writeRank - acrossCoalesced.getRank();
    SmallVector<int64_t> secondReductionDims;
    for (int64_t d = numParallelDims; d < writeRank; ++d) {
      secondReductionDims.push_back(d);
    }

    Value acc = reduceOp.getAcc();
    auto secondReduce = vector::MultiDimReductionOp::create(
        rewriter, loc, kind, toLayout.getResult(), acc, secondReductionDims);

    // Set layouts for redistribution of the second reduce.
    auto accVector = dyn_cast<VectorValue>(acc);
    if (accVector) {
      VectorLayoutInterface accLayout(
          dyn_cast<PackLayoutAttr>(signature[accVector]));
      setSignatureForRedistribution(rewriter, secondReduce,
                                    {readLayoutIface, accLayout}, {accLayout});
    } else {
      setSignatureForRedistribution(rewriter, secondReduce, {readLayoutIface},
                                    {});
    }

    rewriter.replaceOp(reduceOp, secondReduce.getResult());
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

} // namespace

void populateMapDistributeReductionPatterns(RewritePatternSet &patterns,
                                            Value threadId,
                                            int64_t subgroupSize) {
  patterns.add<MapDistributeMultiReduction>(patterns.getContext(), threadId,
                                            subgroupSize);
}

} // namespace mlir::iree_compiler
