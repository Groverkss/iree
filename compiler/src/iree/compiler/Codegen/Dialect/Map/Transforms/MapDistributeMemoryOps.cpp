// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapPatterns.h"

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapDistributionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

using IREE::Map::PackLayoutAttr;
using IREE::VectorExt::DistributionSignature;
using IREE::VectorExt::OpDistributionPattern;
using IREE::VectorExt::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

namespace {

/// Compute per-original-dim value offsets for a given iteration point.
/// All values are static (compile-time constants).
static SmallVector<int64_t> computeValueOffsets(int32_t rank,
                                                ArrayRef<LeafDimInfo> leafMap,
                                                ArrayRef<int64_t> iterOffsets) {
  SmallVector<int64_t> offsets(rank, 0);
  for (size_t i = 0; i < leafMap.size(); ++i) {
    offsets[leafMap[i].origDim] += iterOffsets[i] * leafMap[i].leafStride;
  }
  return offsets;
}

/// Build transfer indices: base[memDim] + threadOffset[vecDim] +
/// valueOffset[vecDim] for each memref dim that the permutation map touches.
static SmallVector<Value> buildTransferIndices(OpBuilder &b, Location loc,
                                               ValueRange baseIndices,
                                               AffineMap permutationMap,
                                               ArrayRef<Value> threadOffsets,
                                               ArrayRef<int64_t> valueOffsets) {
  SmallVector<Value> indices(baseIndices.begin(), baseIndices.end());
  for (auto [vecDim, expr] : llvm::enumerate(permutationMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      continue; // Broadcast dim — index unchanged.
    }
    int64_t memDim = dimExpr.getPosition();
    indices[memDim] =
        arith::AddIOp::create(b, loc, indices[memDim], threadOffsets[vecDim]);
    if (valueOffsets[vecDim] != 0) {
      Value offset =
          arith::ConstantIndexOp::create(b, loc, valueOffsets[vecDim]);
      indices[memDim] = arith::AddIOp::create(b, loc, indices[memDim], offset);
    }
  }
  return indices;
}

//===----------------------------------------------------------------------===//
// MapDistributeTransferRead
//===----------------------------------------------------------------------===//

struct MapDistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  MapDistributeTransferRead(MLIRContext *ctx, Value threadId)
      : OpDistributionPattern(ctx), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // 1. Guard: must be PackLayoutAttr on the result.
    auto layout = dyn_cast<PackLayoutAttr>(signature[readOp.getResult()]);
    if (!layout) {
      return rewriter.notifyMatchFailure(readOp, "not a PackLayout");
    }

    // 2. Guard: memref source, no mask (v1).
    if (!isa<MemRefType>(readOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(readOp,
                                         "distribution expects memrefs");
    }
    if (readOp.getMask()) {
      return rewriter.notifyMatchFailure(readOp,
                                         "masked reads not yet supported");
    }

    Location loc = readOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type elemTy = readOp.getVectorType().getElementType();
    AffineMap permMap = readOp.getPermutationMap();

    // 3. Compute distributed shape and leaf mapping.
    SmallVector<int64_t> distShape =
        cast<VectorLayoutInterface>(cast<Attribute>(layout))
            .getDistributedShape();
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);
    int64_t distRank = distShape.size();

    // 4. Contiguous load width: last distDim if its leaf stride == 1.
    int64_t innerWidth =
        (leafMap.back().leafStride == 1) ? distShape.back() : 1;

    // 5. Tile shape: all 1s except last dim = innerWidth.
    SmallVector<int64_t> tileShape(distRank, 1);
    tileShape.back() = innerWidth;

    // 6. Inner vector type (1D).
    auto innerVecTy = VectorType::get({innerWidth}, elemTy);

    // 7. Inner permutation map: read along the memref dim corresponding to
    //    the last distributed dim's original dim.
    int64_t origDim = leafMap.back().origDim;
    AffineExpr origExpr = permMap.getResult(origDim);
    auto origDimExpr = dyn_cast<AffineDimExpr>(origExpr);
    if (!origDimExpr) {
      return rewriter.notifyMatchFailure(
          readOp, "contiguous dim is broadcast in permutation map");
    }
    AffineMap innerPermMap =
        AffineMap::get(permMap.getNumDims(), 0, {origDimExpr}, ctx);

    // Inner in_bounds: use the original in_bounds for the contiguous dim.
    SmallVector<bool> innerInBounds = {readOp.isDimInBounds(origDim)};

    // 8. Thread offsets (computed once).
    SmallVector<Value> threadOffsets =
        buildThreadOffsets(rewriter, loc, layout, threadId);

    // 9. Zero accumulator.
    auto distVecTy = VectorType::get(distShape, elemTy);
    Value acc = arith::ConstantOp::create(rewriter, loc, distVecTy,
                                          rewriter.getZeroAttr(distVecTy));

    // 10. Iterate over the distributed shape.
    SmallVector<int64_t> strides(/*count=*/1, /*value=*/1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<int64_t> valOffsets =
          computeValueOffsets(layout.getRank(), leafMap, offsets);
      SmallVector<Value> indices =
          buildTransferIndices(rewriter, loc, readOp.getIndices(), permMap,
                               threadOffsets, valOffsets);

      auto slicedRead = vector::TransferReadOp::create(
          rewriter, loc, innerVecTy, readOp.getBase(), indices,
          AffineMapAttr::get(innerPermMap), readOp.getPadding(),
          /*mask=*/Value(), rewriter.getBoolArrayAttr(innerInBounds));

      acc = vector::InsertStridedSliceOp::create(rewriter, loc, slicedRead, acc,
                                                 offsets, strides);
    }

    // 11. Replace.
    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  Value threadId;
};

//===----------------------------------------------------------------------===//
// MapDistributeTransferWrite
//===----------------------------------------------------------------------===//

struct MapDistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  MapDistributeTransferWrite(MLIRContext *ctx, Value threadId)
      : OpDistributionPattern(ctx), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto layout =
        dyn_cast<PackLayoutAttr>(signature[writeOp.getValueToStore()]);
    if (!layout) {
      return rewriter.notifyMatchFailure(writeOp, "not a PackLayout");
    }

    if (!isa<MemRefType>(writeOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "distribution expects memrefs");
    }
    if (writeOp.getMask()) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "masked writes not yet supported");
    }

    Location loc = writeOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    AffineMap permMap = writeOp.getPermutationMap();

    SmallVector<int64_t> distShape =
        cast<VectorLayoutInterface>(cast<Attribute>(layout))
            .getDistributedShape();
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);
    int64_t distRank = distShape.size();

    int64_t innerWidth =
        (leafMap.back().leafStride == 1) ? distShape.back() : 1;

    SmallVector<int64_t> tileShape(distRank, 1);
    tileShape.back() = innerWidth;

    auto innerVecTy =
        VectorType::get({innerWidth}, writeOp.getVectorType().getElementType());

    int64_t origDim = leafMap.back().origDim;
    AffineExpr origExpr = permMap.getResult(origDim);
    auto origDimExpr = dyn_cast<AffineDimExpr>(origExpr);
    if (!origDimExpr) {
      return rewriter.notifyMatchFailure(
          writeOp, "contiguous dim is broadcast in permutation map");
    }
    AffineMap innerPermMap =
        AffineMap::get(permMap.getNumDims(), 0, {origDimExpr}, ctx);

    SmallVector<bool> innerInBounds = {writeOp.isDimInBounds(origDim)};

    SmallVector<Value> threadOffsets =
        buildThreadOffsets(rewriter, loc, layout, threadId);

    Value distributedVec = getDistributed(
        rewriter, cast<TypedValue<VectorType>>(writeOp.getValueToStore()),
        cast<VectorLayoutInterface>(cast<Attribute>(layout)));

    SmallVector<int64_t> extractStrides(distRank, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<int64_t> valOffsets =
          computeValueOffsets(layout.getRank(), leafMap, offsets);
      SmallVector<Value> indices =
          buildTransferIndices(rewriter, loc, writeOp.getIndices(), permMap,
                               threadOffsets, valOffsets);

      Value slice = vector::ExtractStridedSliceOp::create(
          rewriter, loc, distributedVec, offsets, tileShape, extractStrides);
      // Collapse to 1D for the inner write.
      if (distRank > 1) {
        slice = vector::ShapeCastOp::create(rewriter, loc, innerVecTy, slice);
      }

      vector::TransferWriteOp::create(rewriter, loc, slice, writeOp.getBase(),
                                      indices, AffineMapAttr::get(innerPermMap),
                                      /*mask=*/Value(),
                                      rewriter.getBoolArrayAttr(innerInBounds));
    }

    rewriter.eraseOp(writeOp);
    return success();
  }

  Value threadId;
};

//===----------------------------------------------------------------------===//
// MapDistributeToLayout (shared-memory fallback)
//===----------------------------------------------------------------------===//

struct MapDistributeToLayout final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  MapDistributeToLayout(MLIRContext *ctx, Value threadId)
      : OpDistributionPattern(ctx), threadId(threadId) {}

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto input = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    auto oldLayout = dyn_cast<PackLayoutAttr>(signature[input]);
    auto newLayout = dyn_cast<PackLayoutAttr>(signature[output]);
    if (!oldLayout || !newLayout) {
      return failure();
    }
    Location loc = toLayoutOp.getLoc();
    VectorType vecTy = input.getType();

    if (oldLayout.coalesce() == newLayout.coalesce()) {
      auto oldLayoutIface =
          cast<VectorLayoutInterface>(cast<Attribute>(oldLayout));
      auto newLayoutIface =
          cast<VectorLayoutInterface>(cast<Attribute>(newLayout));
      VectorValue distributed = getDistributed(rewriter, input, oldLayoutIface);
      auto newShape = newLayoutIface.getDistributedShape();
      auto newType = VectorType::get(newShape, vecTy.getElementType());
      auto shapeCast =
          vector::ShapeCastOp::create(rewriter, loc, newType, distributed);
      replaceOpWithDistributedValues(rewriter, toLayoutOp,
                                     shapeCast.getResult());
      return success();
    }
    int64_t rank = vecTy.getRank();

    // Allocate shared memory.
    auto addrSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto memrefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType(),
                                    MemRefLayoutAttrInterface{}, addrSpace);
    auto alloc = memref::AllocOp::create(rewriter, loc, memrefTy);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> indices(rank, c0);
    SmallVector<bool> inBounds(rank, true);

    // WAR barrier: if the alloc is hoisted out of a loop, a previous
    // iteration's read may still be in flight.
    gpu::BarrierOp::create(rewriter, loc);

    // Write with old layout.
    auto write = vector::TransferWriteOp::create(rewriter, loc, input, alloc,
                                                 indices, inBounds);
    VectorLayoutInterface oldLayoutIface(oldLayout);
    VectorLayoutInterface newLayoutIface(newLayout);
    setSignatureForRedistribution(rewriter, write, {oldLayoutIface}, {});

    // RAW barrier: ensure all threads finish writing before any reads.
    gpu::BarrierOp::create(rewriter, loc);

    // Read with new layout.
    Value padding = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(vecTy.getElementType()));
    auto identityMap =
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(rank));
    auto read = vector::TransferReadOp::create(
        rewriter, loc, vecTy, alloc, indices, identityMap, padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    setSignatureForRedistribution(rewriter, read, {}, {newLayoutIface});

    rewriter.replaceOp(toLayoutOp, read.getResult());
    return success();
  }

  Value threadId;
};

} // namespace

void populateMapDistributeMemoryOpPatterns(RewritePatternSet &patterns,
                                           Value threadId) {
  patterns.add<MapDistributeTransferRead, MapDistributeTransferWrite,
               MapDistributeToLayout>(patterns.getContext(), threadId);
}

} // namespace mlir::iree_compiler
