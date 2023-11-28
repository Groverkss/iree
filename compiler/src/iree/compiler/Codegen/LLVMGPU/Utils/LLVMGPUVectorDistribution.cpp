// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE_ANCHORS "iree-llvmgpu-vector-distribution-anchors"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

namespace {

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, RewriterBase &rewriter,
                     LayoutProvider *provider, VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), rewriter(rewriter) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;

    root->walk([&](Operation *op) {
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        VectorLayoutInterface layout =
            analysis.getLayout<BlockLayoutAttr>(operand);
        if (layout) {
          op->setAttr(("layout" + std::to_string(index)), layout);
        }
      }
    });
  }

  void distribute() {
    root->walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);

      TypeSwitch<Operation *, void>(op)
          .Case<vector::ContractionOp>([&](auto contractOp) {})
          .Case<vector::TransferReadOp>([&](auto transferReadOp) {
            distributeTransferReads(transferReadOp);
            return;
          })
          .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {})
          .Case<arith::ConstantOp>([&](auto constantOp) {})
          .Default([&](auto op) {});
    });
  }

private:
  void distributeTransferWrites(vector::TransferWriteOp transferWriteOp) {}

  void distributeTransferReads(vector::TransferReadOp transferReadOp) {
    Location loc = transferReadOp.getLoc();
    TypedValue<ShapedType> source = transferReadOp.getSource();
    TypedValue<VectorType> simdVector = transferReadOp.getResult();
    SmallVector<Value> indices = transferReadOp.getIndices();
    Type elType = simdVector.getType().getElementType();
    ArrayRef<int64_t> simdShape = simdVector.getType().getShape();

    // Get layout of the result vector.
    BlockLayoutAttr layout = analysis.getLayout<BlockLayoutAttr>(simdVector);
    if (!layout) {
      return;
    }

    // Vector accumulator: [batchSize, distributedSize]
    SmallVector<int64_t> numBatches = layout.getNumBatches(simdShape);
    ArrayRef<int64_t> numDistributed = layout.getBatch();
    ArrayRef<int64_t> simtVecShape = layout.getThread();
    SmallVector<int64_t> vectorShape;
    vectorShape.append(numBatches.begin(), numBatches.end());
    vectorShape.append(numDistributed.begin(), numDistributed.end());
    vectorShape.append(simtVecShape.begin(), simtVecShape.end());
    VectorType vecType = VectorType::get(vectorShape, elType);
    Value vector = rewriter.create<arith::ConstantOp>(
        loc, vecType, rewriter.getZeroAttr(vecType));

    // Iterate over all batches.
    layout.forAllBatchTiles(simdShape, [&](ArrayRef<int64_t> batch) {
      // Iterate over all distributed tiles in the batch.
      layout.forAllDistributedTiles([&](ArrayRef<int64_t> distributed) {
        // This tile will be distributed over threads.
        // Iterate over all elements in the thread tile.
        layout.forAllElementsInThreadTile([&](ArrayRef<int64_t> elms) {
          SmallVector<Value> newIndices =
              getDistributedIndices(loc, indices, batch, distributed, elms,
                                    transferReadOp.getPermutationMap(), layout);
          Value load = rewriter.create<memref::LoadOp>(loc, source, newIndices);
          VectorType loadedVectorType = VectorType::get({1}, elType);
          Value loadedVector =
              rewriter.create<vector::BroadcastOp>(loc, loadedVectorType, load);
          // vector.insert_strided_slice offsets: [batch, distributed, elms]
          SmallVector<int64_t> offsets;
          offsets.append(batch.begin(), batch.end());
          offsets.append(distributed.begin(), distributed.end());
          offsets.append(elms.begin(), elms.end());
          SmallVector<int64_t> strides{1};
          vector = rewriter.create<vector::InsertStridedSliceOp>(
              loc, loadedVector, vector, offsets, strides);
        });
      });
    });

    simdToSimt.map(transferReadOp.getResult(), vector);
  }

  void distributeContractions(vector::ContractionOp contractionOp) {}

  /// Get indices of transfer op after distribution.
  SmallVector<Value>
  getDistributedIndices(Location loc, ArrayRef<Value> indices,
                        ArrayRef<int64_t> batch, ArrayRef<int64_t> distributed,
                        ArrayRef<int64_t> thread, AffineMap permutationMap,
                        BlockLayoutAttr layout) {
    // Get tiles.
    ArrayRef<int64_t> batchTile = layout.getBatch();
    ArrayRef<int64_t> distributedTile = layout.getDistributed();
    ArrayRef<int64_t> threadTile = layout.getThread();

    auto computeDim = [&](int64_t dim) {
      // thread +
      // gpuIdx * threadTile +
      // distributedIdx * (distributedTile * threadTile) +
      // batchIdx * (batchTile * distributedTile * threadTile)
      AffineExpr threadIdx = rewriter.getAffineDimExpr(dim);
      return (thread[dim]) + (threadIdx * threadTile[dim]) +
             (distributed[dim] * distributedTile[dim] * threadTile[dim]) +
             (batch[dim] * batchTile[dim] * distributedTile[dim] *
              threadTile[dim]);
    };

    SmallVector<Value> offsets(batch.size());
    for (int64_t i = 0; i < batch.size(); ++i) {
      AffineMap map =
          AffineMap::get(3, 0, computeDim(i), rewriter.getContext());
      offsets[i] =
          rewriter.create<affine::AffineApplyOp>(loc, map, getThreadIds());
    }

    SmallVector<Value> newIndices{indices.begin(), indices.end()};
    int64_t laneDim = 0;
    for (AffineExpr expr : permutationMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        continue;
      unsigned pos = dimExpr.getPosition();
      newIndices[pos] = rewriter.create<arith::AddIOp>(loc, offsets[laneDim++],
                                                       newIndices[pos]);
    }
    return newIndices;
  }

  ArrayRef<Value> getThreadIds() {
    if (threadIds.empty()) {
      threadIds = {
          rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::x),
          rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::y),
          rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::z)};
    }
    return threadIds;
  }

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  RewriterBase &rewriter;
  IRMapping simdToSimt;
  SmallVector<Value> threadIds;
}; // namespace

} // namespace

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  NVIDIALayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
