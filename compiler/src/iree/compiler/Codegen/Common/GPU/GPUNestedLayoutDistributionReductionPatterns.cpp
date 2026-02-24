// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

namespace {

int64_t getShuffleOffset(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadStrides()[dim];
}

int64_t getShuffleWidth(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadTile()[dim];
}

/// Thread-level butterfly reduction via gpu.subgroup_reduce.
/// For each element in the flat vector, reduce across all reduction dimensions
/// using clustered subgroup reduces with the appropriate stride and width.
FailureOr<VectorValue> butterflyThreadReduction(RewriterBase &rewriter,
                                                NestedLayoutAttr layout,
                                                VectorValue flat,
                                                vector::CombiningKind kind,
                                                ArrayRef<bool> reductionMask) {
  VectorType flatVecType = flat.getType();
  int64_t numElements = flatVecType.getNumElements();
  Location loc = flat.getLoc();

  auto constOp = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(flatVecType));
  auto res = cast<VectorValue>(constOp.getResult());

  for (unsigned i = 0; i < numElements; ++i) {
    Value extracted = vector::ExtractOp::create(rewriter, loc, flat, i);
    for (unsigned j = 0, e = reductionMask.size(); j != e; ++j) {
      if (reductionMask[j]) {
        int64_t offset = getShuffleOffset(layout, j);
        int64_t width = getShuffleWidth(layout, j);
        assert(offset <= std::numeric_limits<uint32_t>::max() &&
               width <= std::numeric_limits<uint32_t>::max());
        extracted = gpu::SubgroupReduceOp::create(
            rewriter, loc, extracted, combiningKindToAllReduce(kind),
            /*uniform=*/false, /*cluster_size=*/width,
            /*cluster_stride=*/offset);
      }
    }
    res = vector::InsertOp::create(rewriter, loc, extracted, res, i);
  }
  return res;
}

/// Compute a layout for reading back from shared memory after a subgroup
/// write. The subgroup tile along the given dims is redistributed to threads.
NestedLayoutAttr getLayoutForSubgroupFromBuffer(NestedLayoutAttr srcLayout,
                                                ArrayRef<int64_t> dims) {
  auto subgroupTileLens =
      llvm::to_vector_of<int64_t>(srcLayout.getSubgroupTile());
  auto batchTileLens = llvm::to_vector_of<int64_t>(srcLayout.getBatchTile());
  auto outerTileLens = llvm::to_vector_of<int64_t>(srcLayout.getOuterTile());
  auto threadTileLens = llvm::to_vector_of<int64_t>(srcLayout.getThreadTile());
  auto elementTileLens =
      llvm::to_vector_of<int64_t>(srcLayout.getElementTile());
  auto subgroupStrides =
      llvm::to_vector_of<int64_t>(srcLayout.getSubgroupStrides());
  auto threadStrides =
      llvm::to_vector_of<int64_t>(srcLayout.getThreadStrides());

  // Check if we had enough threads on one of the reduction dimensions
  // to use for a subgroup reduction. If not, do a serialized reduction.
  int64_t threadsRequired = 1;
  for (int64_t rDim : dims) {
    threadsRequired *= llvm::PowerOf2Ceil(subgroupTileLens[rDim]);
  }
  std::optional<int64_t> availableThreads;
  int64_t threadStride = 0;
  for (int64_t rDim : dims) {
    if (threadTileLens[rDim] >= threadsRequired) {
      availableThreads = threadTileLens[rDim];
      threadStride = threadStrides[rDim];
      break;
    }
  }

  for (int64_t rDim : dims) {
    batchTileLens[rDim] = 1;
    outerTileLens[rDim] = 1;
    elementTileLens[rDim] = 1;
    if (availableThreads.has_value()) {
      int64_t used = llvm::PowerOf2Ceil(subgroupTileLens[rDim]);
      threadStrides[rDim] = threadStride;
      threadTileLens[rDim] = used;
      *availableThreads /= used;
      threadStride *= used;
    } else {
      threadStrides[rDim] = 0;
      threadTileLens[rDim] = 1;
    }
    subgroupTileLens[rDim] = 1;
    subgroupStrides[rDim] = 0;
  }
  return NestedLayoutAttr::get(srcLayout.getContext(), subgroupTileLens,
                               batchTileLens, outerTileLens, threadTileLens,
                               elementTileLens, subgroupStrides, threadStrides);
}

/// Mixin that provides shared reduction/scan helpers needing access to
/// setSignatureForRedistribution (which is protected on DistributionPattern).
/// PatternBase should be OpDistributionPattern<Op> or
/// MaskedOpDistributionPattern<Op>.
template <typename PatternBase>
struct ReductionDistributionBase : PatternBase {
  using PatternBase::PatternBase;

  /// Write an iso-rank-reshaped distributed vector to shared memory.
  /// Handles: ToSIMD, alloc, barrier, write with layout annotation, barrier.
  /// Returns the allocated shared memory buffer.
  Value writePartialToSharedMemory(RewriterBase &rewriter, Location loc,
                                   VectorValue isoRankDistributed,
                                   ArrayRef<int64_t> partialReductionShape,
                                   NestedLayoutAttr srcLayout,
                                   ArrayRef<int64_t> reducedDims,
                                   Type elemTy) const {
    int64_t rank = srcLayout.getRank();

    // ToSIMD to undistributed form.
    auto unDistType = VectorType::get(partialReductionShape, elemTy);
    VectorValue valueToWrite =
        ToSIMDOp::create(rewriter, loc, unDistType, isoRankDistributed);

    // Allocate shared memory + barrier.
    auto workgroupAS = Attribute(gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup));
    MemRefType allocType = MemRefType::get(partialReductionShape, elemTy,
                                           AffineMap(), workgroupAS);
    auto alloc = memref::AllocOp::create(rewriter, loc, allocType);
    gpu::BarrierOp::create(rewriter, loc, alloc);

    // Write with layout annotation.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> indices(rank, c0);
    SmallVector<bool> inBounds(rank, true);
    auto write = vector::TransferWriteOp::create(rewriter, loc, valueToWrite,
                                                 alloc, indices, inBounds);
    auto subgroupTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupTile());
    auto batchTileLens = llvm::to_vector_of<int64_t>(srcLayout.getBatchTile());
    auto outerTileLens = llvm::to_vector_of<int64_t>(srcLayout.getOuterTile());
    auto threadTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadTile());
    auto elementTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getElementTile());
    auto subgroupStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupStrides());
    auto threadStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadStrides());
    for (int64_t rDim : reducedDims) {
      batchTileLens[rDim] = 1;
      outerTileLens[rDim] = 1;
      threadTileLens[rDim] = 1;
      elementTileLens[rDim] = 1;
      threadStrides[rDim] = 0;
    }
    auto writeLayout = NestedLayoutAttr::get(
        rewriter.getContext(), subgroupTileLens, batchTileLens, outerTileLens,
        threadTileLens, elementTileLens, subgroupStrides, threadStrides);
    this->setSignatureForRedistribution(rewriter, write, {writeLayout}, {});

    // Barrier for write completion.
    gpu::BarrierOp::create(rewriter, loc, alloc);
    return alloc;
  }

  /// Perform inter-subgroup reduction via shared memory.
  /// 1. Reshape threadReduced to iso-rank (reduction dims -> 1)
  /// 2. Write to shared memory with layout annotation
  /// 3. Read back with redistribution layout
  /// 4. Second MultiDimReduction to complete the reduction
  /// Pass acc=Value() to use identity accumulator (for AssociativeReduce).
  Value doSubgroupReduction(RewriterBase &rewriter, Location loc,
                            VectorValue srcVector, NestedLayoutAttr srcLayout,
                            ArrayRef<int64_t> reductionDims,
                            VectorValue threadReduced,
                            vector::CombiningKind kind, Value acc,
                            VectorLayoutInterface resLayout) const {
    int64_t rank = srcLayout.getRank();
    Type elemTy = srcVector.getType().getElementType();

    // Reshape to iso-rank: set reduction dims to 1 in all 3 tiers.
    SmallVector<int64_t> partialDistShape = srcLayout.getDistributedShape();
    for (int64_t tier : llvm::seq<int64_t>(3)) {
      for (int64_t rDim : reductionDims) {
        partialDistShape[tier * rank + rDim] = 1;
      }
    }
    VectorType partialDistType = VectorType::get(partialDistShape, elemTy);
    VectorValue isoRank = vector::ShapeCastOp::create(
        rewriter, loc, partialDistType, threadReduced);

    // Compute undistributed partial reduction shape.
    SmallVector<int64_t> preDistrShape =
        srcLayout.getUndistributedPackedShape();
    SmallVector<int64_t> partialReductionShape =
        llvm::to_vector(srcVector.getType().getShape());
    for (int64_t rDim : reductionDims) {
      partialReductionShape[rDim] = preDistrShape[rDim];
    }

    // Write to shared memory.
    Value buffer = writePartialToSharedMemory(rewriter, loc, isoRank,
                                              partialReductionShape, srcLayout,
                                              reductionDims, elemTy);

    // Read from buffer with layout that redistributes subgroup tile to threads.
    NestedLayoutAttr readLayout =
        getLayoutForSubgroupFromBuffer(srcLayout, reductionDims);
    Value padValue = getCombiningIdentityValue(loc, rewriter, kind,
                                               getElementTypeOrSelf(buffer));
    auto readTy = VectorType::get(readLayout.getUndistributedShape(),
                                  getElementTypeOrSelf(buffer));
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(readLayout.getRank(), true));
    auto mask = vector::CreateMaskOp::create(
        rewriter, loc, readTy.clone(rewriter.getI1Type()),
        memref::getMixedSizes(rewriter, loc, buffer));
    auto read = vector::TransferReadOp::create(
        rewriter, loc, readTy, buffer,
        SmallVector<Value>(readLayout.getRank(), zero),
        rewriter.getMultiDimIdentityMap(readLayout.getRank()), padValue, mask,
        inBounds);
    this->setSignatureForRedistribution(rewriter, mask, {}, {readLayout});
    this->setSignatureForRedistribution(rewriter, read, {readLayout},
                                        {readLayout});

    // Second reduction to complete inter-subgroup reduce.
    Value reductionAcc = acc;
    if (!reductionAcc) {
      reductionAcc = getCombiningIdentityValue(loc, rewriter, kind, readTy);
    }
    auto secondReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, kind, read, reductionAcc, reductionDims);
    if (resLayout) {
      this->setSignatureForRedistribution(rewriter, secondReduction,
                                          {readLayout, resLayout}, {resLayout});
    } else {
      this->setSignatureForRedistribution(rewriter, secondReduction,
                                          {readLayout}, {});
    }
    return secondReduction.getResult();
  }
};

/// The lowering for multi_reduction is done in the following steps:
///   1. Local Reduce: Each thread reduces all elements carried by it along
///      the reduction dimensions. This is the batch, outer and element dims.
///   2. Thread Reduce: Each thread reduces result of step 1 across threads
///      by doing a butterfly shuffle.
///   3. Accumulator Reduce: Each thread reduces its intermediate reduced
///      results with the accumulator it holds.
///   4. Subgroup reduce: Each subgroup stores partial reductions to shared
///      memory, which are reloaded into a layout where partial reductions
///      are placed inside threads.
struct DistributeMultiReduction final
    : ReductionDistributionBase<
          MaskedOpDistributionPattern<vector::MultiDimReductionOp>> {
  using ReductionDistributionBase::ReductionDistributionBase;

  DistributeMultiReduction(MLIRContext *context, int64_t /*subgroupSize*/,
                           int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : ReductionDistributionBase(context, benefit),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp multiReduceOp,
                  DistributionSignature &signature, vector::MaskOp maskOp,
                  std::optional<DistributionSignature> &maskSignature,
                  PatternRewriter &rewriter) const {
    Location loc = multiReduceOp.getLoc();
    VectorValue srcVector = multiReduceOp.getSource();
    Value acc = multiReduceOp.getAcc();
    Value res = multiReduceOp.getResult();
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);

    auto srcLayout =
        dyn_cast_if_present<NestedLayoutAttr>(signature[srcVector]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(multiReduceOp,
                                         "expected nested layout attr");
    }

    Type elemTy = srcVector.getType().getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth > maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, llvm::formatv("element bitwidth ({0}) greater than "
                                       "maxBitsPerShuffle ({1})",
                                       elemBitwidth, maxBitsPerShuffle));
    }

    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);

    Value disAcc;
    if (accVector) {
      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
    } else {
      // Scalars are always distributed to all threads already.
      disAcc = multiReduceOp.getAcc();
    }

    VectorValue mask = nullptr;
    if (maskOp) {
      auto maskLayout = dyn_cast_if_present<NestedLayoutAttr>(
          maskSignature.value()[maskOp.getMask()]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(maskOp,
                                           "expected nested layout attr");
      }
      mask = getDistributed(rewriter, maskOp.getMask(), maskLayout);
      Value passThruSrc = getCombiningIdentityValue(
          loc, rewriter, multiReduceOp.getKind(), disSrc.getType());

      disSrc = cast<VectorValue>(
          arith::SelectOp::create(rewriter, loc, mask, disSrc, passThruSrc)
              .getResult());
    }

    SmallVector<bool> reducedDims = multiReduceOp.getReductionMask();
    int64_t rank = srcVector.getType().getRank();

    // Do thread local reduce.

    // The distributed reduction mask is simply the same mask appended
    // thrice.
    SmallVector<bool> distributedReductionMask;
    distributedReductionMask.reserve(3 * rank);
    for (int i = 0; i < 3; ++i) {
      distributedReductionMask.append(reducedDims.begin(), reducedDims.end());
    }
    Value localInit = getCombiningIdentityValue(
        loc, rewriter, multiReduceOp.getKind(), disAcc.getType());
    Value localReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, disSrc, localInit, distributedReductionMask,
        multiReduceOp.getKind());

    // TODO: As per current upstream lowering implementations, there is no point
    // in doing this because it does a select much later in a finer granularity
    // rather than supporting predication. Moreover, since we are doing a select
    // to cater reductions accross the distribution, we can choose not to mask
    // the op post-distribution.

    VectorValue locallyReduced;
    if (accVector) {
      locallyReduced = dyn_cast<VectorValue>(localReduction);
    } else {
      // Broadcast scalar accumulator to vector.
      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, elemTy);
      locallyReduced =
          vector::BroadcastOp::create(rewriter, loc, vecType, localReduction);
    }

    assert(locallyReduced && "result should have been a vector");

    // Flatten the locally reduced value.
    VectorValue threadReduced = locallyReduced;
    VectorType shaped = locallyReduced.getType();
    bool hasThreadReductions =
        llvm::any_of(multiReduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getThreadTile()[rDim] > 1;
        });
    if (hasThreadReductions) {
      int64_t numElements = shaped.getNumElements();
      SmallVector<int64_t> flatShape(1, numElements);
      VectorType flatVecType = VectorType::get(flatShape, elemTy);
      VectorValue flat = vector::ShapeCastOp::create(rewriter, loc, flatVecType,
                                                     locallyReduced);

      // Do inter-thread/warp reduce.
      FailureOr<VectorValue> threadReducedFlat = butterflyThreadReduction(
          rewriter, srcLayout, flat, multiReduceOp.getKind(), reducedDims);
      if (failed(threadReducedFlat)) {
        return rewriter.notifyMatchFailure(multiReduceOp,
                                           "thread reduction failed");
      }

      // Do reduction against accumulator, which needs to be done after thread
      // reduction.
      threadReduced = vector::ShapeCastOp::create(rewriter, loc, shaped,
                                                  threadReducedFlat.value());
    }

    if (!accVector) {
      // Broadcast the scalar (e.g., f32) to a vector type (e.g., vector<f32>)
      // because the following implementation requires the operand to be a
      // vector.
      disAcc = vector::BroadcastOp::create(rewriter, loc, shaped, disAcc);
    }

    bool hasSubgroupReductions =
        llvm::any_of(multiReduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getSubgroupTile()[rDim] > 1;
        });
    // No subgroup reductions -- finalize with accumulator and exit.
    if (!hasSubgroupReductions) {
      Value accReduction = vector::makeArithReduction(
          rewriter, loc, multiReduceOp.getKind(), threadReduced, disAcc);
      auto accReduced = dyn_cast<VectorValue>(accReduction);
      if (!accReduced) {
        return rewriter.notifyMatchFailure(
            multiReduceOp, "expected vector result from acc reduction");
      }
      if (resVector) {
        replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
      } else {
        Value accReducedVal = vector::ExtractOp::create(
            rewriter, loc, accReduction, ArrayRef{int64_t(0)});
        replaceOpWithDistributedValues(rewriter, multiReduceOp, accReducedVal);
      }
      return success();
    }
    // do inter-subgroup reductions
    Value subgroupReduced = doSubgroupReduction(
        rewriter, loc, srcVector, srcLayout, multiReduceOp.getReductionDims(),
        threadReduced, multiReduceOp.getKind(), acc, signature[resVector]);
    rewriter.replaceOp(multiReduceOp, subgroupReduced);
    return success();
  }

  int64_t maxBitsPerShuffle;
};

/// Try to match a combiner region with a single operand group (N=1)
/// to a CombiningKind. Returns std::nullopt if the combiner is not
/// a simple arith op or if the region has multiple operand groups.
std::optional<vector::CombiningKind> matchCombinerRegion(Region &combiner) {
  Block &body = combiner.front();
  // Only handle single-group combiners (2 args: lhs, rhs).
  if (body.getNumArguments() != 2) {
    return std::nullopt;
  }
  auto &ops = body.getOperations();
  // Expect exactly one binary op + yield.
  if (ops.size() != 2) {
    return std::nullopt;
  }
  return linalg::getCombinerOpKind(&ops.front());
}

/// Distribution pattern for iree_vector_ext.associative_reduce.
///
/// For combiners that match a CombiningKind (single-group, simple arith op),
/// the reduction is lowered identically to DistributeMultiReduction:
///   1. Local reduce via vector.multi_reduction
///   2. Thread reduce via gpu.subgroup_reduce
///   3. Subgroup reduce via shared memory
///
/// For unmatched/variadic combiners, the reduction uses:
///   1. Local reduce via a distributed AssociativeReduceOp
///   2. Thread reduce via explicit gpu.shuffle xor + stack + dim-2 reduce
///   3. Subgroup reduce via shared memory + second AssociativeReduceOp
struct DistributeAssociativeReduce final
    : ReductionDistributionBase<OpDistributionPattern<AssociativeReduceOp>> {
  using ReductionDistributionBase::ReductionDistributionBase;

  DistributeAssociativeReduce(MLIRContext *context, int64_t /*subgroupSize*/,
                              int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : ReductionDistributionBase(context, benefit),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult matchAndRewrite(AssociativeReduceOp reduceOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();
    OperandRange inputs = reduceOp.getInputs();

    auto srcLayout = dyn_cast_if_present<NestedLayoutAttr>(
        signature[cast<VectorValue>(inputs.front())]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "expected nested layout attr");
    }

    Type elemTy = cast<VectorType>(inputs.front().getType()).getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth > maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          reduceOp, "element bitwidth greater than maxBitsPerShuffle");
    }

    // Try to match the combiner to a CombiningKind.
    auto kind = matchCombinerRegion(reduceOp.getCombiner());
    if (!kind) {
      return rewriter.notifyMatchFailure(
          reduceOp, "unmatched/variadic combiners not yet supported");
    }

    // For matched combiners with a single input, delegate to the same
    // strategy as DistributeMultiReduction.
    if (inputs.size() != 1) {
      return rewriter.notifyMatchFailure(
          reduceOp, "matched combiner requires single input group");
    }

    VectorValue srcVector = cast<VectorValue>(inputs.front());
    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);

    SmallVector<bool> reducedDims = reduceOp.getReductionMask();
    int64_t rank = srcVector.getType().getRank();

    // Build the tripled reduction mask for the distributed shape.
    SmallVector<bool> distributedReductionMask;
    distributedReductionMask.reserve(3 * rank);
    for (int i = 0; i < 3; ++i) {
      distributedReductionMask.append(reducedDims.begin(), reducedDims.end());
    }

    // Local reduce via vector.multi_reduction with identity init.
    VectorType disSrcType = disSrc.getType();
    // Compute distributed result type by projecting out reduction dims.
    SmallVector<int64_t> disResultShape;
    for (auto [i, dim] : llvm::enumerate(disSrcType.getShape())) {
      if (!distributedReductionMask[i]) {
        disResultShape.push_back(dim);
      }
    }
    VectorType disResultType = VectorType::get(disResultShape, elemTy);
    Value localInit =
        getCombiningIdentityValue(loc, rewriter, *kind, disResultType);
    Value localReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, disSrc, localInit, distributedReductionMask, *kind);

    VectorValue locallyReduced = cast<VectorValue>(localReduction);

    // Thread reduce (butterfly shuffle).
    VectorValue threadReduced = locallyReduced;
    VectorType shaped = locallyReduced.getType();
    bool hasThreadReductions =
        llvm::any_of(reduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getThreadTile()[rDim] > 1;
        });
    if (hasThreadReductions) {
      int64_t numElements = shaped.getNumElements();
      VectorType flatVecType = VectorType::get({numElements}, elemTy);
      VectorValue flat = vector::ShapeCastOp::create(rewriter, loc, flatVecType,
                                                     locallyReduced);

      FailureOr<VectorValue> threadReducedFlat = butterflyThreadReduction(
          rewriter, srcLayout, flat, *kind, reducedDims);
      if (failed(threadReducedFlat)) {
        return rewriter.notifyMatchFailure(reduceOp, "thread reduction failed");
      }

      threadReduced = vector::ShapeCastOp::create(rewriter, loc, shaped,
                                                  *threadReducedFlat);
    }

    // Subgroup reduce.
    bool hasSubgroupReductions =
        llvm::any_of(reduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getSubgroupTile()[rDim] > 1;
        });
    if (!hasSubgroupReductions) {
      replaceOpWithDistributedValues(rewriter, reduceOp, threadReduced);
      return success();
    }

    // Subgroup reduction via shared memory.
    VectorValue resVector = cast<VectorValue>(reduceOp.getResults().front());
    Value subgroupReduced = doSubgroupReduction(
        rewriter, loc, srcVector, srcLayout, reduceOp.getReductionDims(),
        threadReduced, *kind, /*acc=*/Value(), signature[resVector]);
    rewriter.replaceOp(reduceOp, subgroupReduced);
    return success();
  }

  int64_t maxBitsPerShuffle;
};

/// Populate an AssociativeScanOp's combiner region by cloning a source body.
/// Creates a block with arguments, clones body ops, and adds a yield.
void populateCombinerRegion(PatternRewriter &rewriter, Location loc,
                            Region &region, Block &srcBody, int64_t numGroups,
                            Type elemTy) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Ensure the region is empty before creating our block.
  if (!region.empty()) {
    region.front().erase();
  }
  Block *block = rewriter.createBlock(&region);
  for (int64_t g = 0; g < 2 * numGroups; ++g) {
    block->addArgument(elemTy, loc);
  }
  IRMapping mapping;
  for (int64_t g = 0; g < 2 * numGroups; ++g) {
    mapping.map(srcBody.getArgument(g), block->getArgument(g));
  }
  for (Operation &op : srcBody.without_terminator()) {
    rewriter.clone(op, mapping);
  }
  auto srcYield = cast<IREE::VectorExt::YieldOp>(srcBody.getTerminator());
  SmallVector<Value> yieldedValues;
  for (Value v : srcYield.getOperands()) {
    yieldedValues.push_back(mapping.lookupOrDefault(v));
  }
  IREE::VectorExt::YieldOp::create(rewriter, loc, yieldedValues);
}

/// Clone a combiner region to produce a scalar combine of lhs and rhs.
/// Returns the N yielded results.
SmallVector<Value> cloneCombinerScalar(PatternRewriter &rewriter, Location loc,
                                       Block &body, ArrayRef<Value> lhs,
                                       ArrayRef<Value> rhs) {
  IRMapping mapping;
  int64_t numGroups = lhs.size();
  for (int64_t g = 0; g < numGroups; ++g) {
    mapping.map(body.getArgument(g), lhs[g]);
    mapping.map(body.getArgument(numGroups + g), rhs[g]);
  }
  for (Operation &op : body.without_terminator()) {
    rewriter.clone(op, mapping);
  }
  auto yieldOp = cast<IREE::VectorExt::YieldOp>(body.getTerminator());
  SmallVector<Value> results;
  for (int64_t g = 0; g < numGroups; ++g) {
    results.push_back(mapping.lookupOrDefault(yieldOp.getOperand(g)));
  }
  return results;
}

/// Clone a combiner region to produce a vector combine of lhs and rhs.
/// Body ops are cloned with result types widened to vectors.
/// Assumption: all combiner body ops are element-wise (e.g., arith ops) and
/// can be vectorized by widening result types.
/// TODO: Unify with applyCombinerToVectors in VectorExtOps.cpp.
SmallVector<Value> cloneCombinerVectorized(PatternRewriter &rewriter,
                                           Location loc, Block &body,
                                           ArrayRef<Value> lhs,
                                           ArrayRef<Value> rhs,
                                           ArrayRef<int64_t> resultShape) {
  IRMapping mapping;
  int64_t numGroups = lhs.size();
  for (int64_t g = 0; g < numGroups; ++g) {
    mapping.map(body.getArgument(g), lhs[g]);
    mapping.map(body.getArgument(numGroups + g), rhs[g]);
  }
  for (Operation &op : body.without_terminator()) {
    Operation *cloned = rewriter.clone(op, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), cloned->getResults())) {
      VectorType vecTy = VectorType::get(resultShape, oldResult.getType());
      newResult.setType(vecTy);
      mapping.map(oldResult, newResult);
    }
  }
  auto yieldOp = cast<IREE::VectorExt::YieldOp>(body.getTerminator());
  SmallVector<Value> results;
  for (int64_t g = 0; g < numGroups; ++g) {
    results.push_back(mapping.lookupOrDefault(yieldOp.getOperand(g)));
  }
  return results;
}

/// Distribution pattern for iree_vector_ext.associative_scan.
///
/// Unlike reduce (which removes dims), scan preserves the input shape.
/// Uses Hillis-Steele prefix scan via gpu.shuffle up for thread-level
/// communication.
///
/// For each scan dim d:
///   1. Local scan: transpose + merge scan tiers + AssociativeScanOp + split
///   2. Thread scan: extract thread total, Hillis-Steele shuffle up, broadcast
///   3. Subgroup scan: shared memory redistribution + AssociativeScanOp
struct DistributeAssociativeScan final
    : ReductionDistributionBase<OpDistributionPattern<AssociativeScanOp>> {
  using ReductionDistributionBase::ReductionDistributionBase;

  DistributeAssociativeScan(MLIRContext *context, int64_t /*subgroupSize*/,
                            int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : ReductionDistributionBase(context, benefit),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult matchAndRewrite(AssociativeScanOp scanOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = scanOp.getLoc();
    OperandRange inputs = scanOp.getInputs();
    int64_t numGroups = scanOp.getNumOperandGroups();

    auto srcLayout = dyn_cast_if_present<NestedLayoutAttr>(
        signature[cast<VectorValue>(inputs.front())]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(scanOp, "expected nested layout attr");
    }

    Type elemTy = cast<VectorType>(inputs.front().getType()).getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth > maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          scanOp, "element bitwidth greater than maxBitsPerShuffle");
    }

    SmallVector<bool> scanMask = scanOp.getReductionMask();
    int64_t rank = scanOp.getInputRank();
    Block &body = scanOp.getCombiner().front();

    // Pre-flight: reject scan dims that need subgroup-level distribution,
    // which is not yet implemented. This check must happen before any IR
    // mutation to avoid partial rewrites.
    for (int64_t d = 0; d < rank; ++d) {
      if (scanMask[d] && srcLayout.getSubgroupTile()[d] > 1) {
        return rewriter.notifyMatchFailure(
            scanOp, "subgroup scan distribution not yet implemented");
      }
    }

    // Distribute all inputs.
    SmallVector<VectorValue> distributed(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      VectorValue src = cast<VectorValue>(inputs[g]);
      distributed[g] = getDistributed(rewriter, src, signature[src]);
    }

    // Process each scan dim independently.
    for (int64_t d = 0; d < rank; ++d) {
      if (!scanMask[d]) {
        continue;
      }

      // Stage 1: Local inclusive scan.
      distributed = doLocalScan(rewriter, loc, srcLayout, distributed, body, d,
                                rank, numGroups);

      // Stage 2: Thread prefix scan.
      if (srcLayout.getThreadTile()[d] > 1) {
        distributed = doThreadScan(rewriter, loc, srcLayout, distributed, body,
                                   d, rank, numGroups);
      }
    }

    SmallVector<Value> distributedValues(distributed.begin(),
                                         distributed.end());
    replaceOpWithDistributedValues(rewriter, scanOp, distributedValues);
    return success();
  }

private:
  /// Stage 1: Local inclusive scan along scan dim d.
  /// Transpose to group scan positions at end, merge, emit AssociativeScanOp,
  /// split back, transpose with inverse perm.
  SmallVector<VectorValue> doLocalScan(PatternRewriter &rewriter, Location loc,
                                       NestedLayoutAttr srcLayout,
                                       ArrayRef<VectorValue> inputs,
                                       Block &body, int64_t d, int64_t rank,
                                       int64_t numGroups) const {
    VectorType disType = inputs.front().getType();
    ArrayRef<int64_t> disShape = disType.getShape();
    Type elemTy = disType.getElementType();

    // The 3 scan-dim positions in distributed shape: d, rank+d, 2*rank+d
    // (batch, outer, element tiers).
    SmallVector<int64_t> scanPositions = {d, rank + d, 2 * rank + d};

    // Build transpose perm: non-scan dims first, then scan dims at end.
    SmallVector<int64_t> perm;
    for (int64_t i = 0; i < 3 * rank; ++i) {
      if (i != d && i != rank + d && i != 2 * rank + d) {
        perm.push_back(i);
      }
    }
    perm.append(scanPositions.begin(), scanPositions.end());

    // Apply transpose.
    SmallVector<int64_t> transposedShape(3 * rank);
    for (int64_t i = 0; i < 3 * rank; ++i) {
      transposedShape[i] = disShape[perm[i]];
    }
    VectorType transposedType = VectorType::get(transposedShape, elemTy);

    SmallVector<VectorValue> transposed(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      transposed[g] = vector::TransposeOp::create(rewriter, loc, transposedType,
                                                  inputs[g], perm);
    }

    // Merge last 3 dims into one.
    int64_t mergedDimSize = transposedShape[3 * rank - 3] *
                            transposedShape[3 * rank - 2] *
                            transposedShape[3 * rank - 1];
    SmallVector<int64_t> mergedShape(transposedShape.begin(),
                                     transposedShape.end() - 3);
    mergedShape.push_back(mergedDimSize);
    VectorType mergedType = VectorType::get(mergedShape, elemTy);

    SmallVector<VectorValue> merged(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      merged[g] =
          vector::ShapeCastOp::create(rewriter, loc, mergedType, transposed[g]);
    }

    // Emit AssociativeScanOp along the merged dim (last dim).
    int64_t mergedRank = mergedShape.size();
    SmallVector<Type> resultTypes(numGroups, mergedType);
    SmallVector<Value> mergedValues(merged.begin(), merged.end());
    auto localScan = AssociativeScanOp::create(
        rewriter, loc, resultTypes, mergedValues,
        rewriter.getDenseI64ArrayAttr({mergedRank - 1}));
    // Populate the combiner region by cloning the source body.
    populateCombinerRegion(rewriter, loc, localScan.getCombiner(), body,
                           numGroups, elemTy);

    SmallVector<VectorValue> scanned(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      scanned[g] = cast<VectorValue>(localScan.getResult(g));
    }

    // ShapeCast back to transposed shape.
    SmallVector<VectorValue> splitBack(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      splitBack[g] = vector::ShapeCastOp::create(rewriter, loc, transposedType,
                                                 scanned[g]);
    }

    // Inverse transpose.
    SmallVector<int64_t> inversePerm(3 * rank);
    for (int64_t i = 0; i < 3 * rank; ++i) {
      inversePerm[perm[i]] = i;
    }
    SmallVector<VectorValue> result(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      result[g] = vector::TransposeOp::create(rewriter, loc, disType,
                                              splitBack[g], inversePerm);
    }
    return result;
  }

  /// Stage 2: Thread prefix scan via Hillis-Steele shuffle up.
  SmallVector<VectorValue> doThreadScan(PatternRewriter &rewriter, Location loc,
                                        NestedLayoutAttr srcLayout,
                                        ArrayRef<VectorValue> inputs,
                                        Block &body, int64_t d, int64_t rank,
                                        int64_t numGroups) const {
    VectorType disType = inputs.front().getType();
    ArrayRef<int64_t> disShape = disType.getShape();
    Type elemTy = disType.getElementType();

    int64_t threadTile = srcLayout.getThreadTile()[d];
    int64_t threadStride = srcLayout.getThreadStrides()[d];
    int64_t width = threadTile * threadStride;

    // Extract thread total: last element along scan dim in the local result.
    int64_t batchSize = disShape[d];
    int64_t outerSize = disShape[rank + d];
    int64_t elemSize = disShape[2 * rank + d];

    // Build the shape for the thread total: all scan-tier dims set to 1.
    SmallVector<int64_t> totalShape(disShape);
    totalShape[d] = 1;
    totalShape[rank + d] = 1;
    totalShape[2 * rank + d] = 1;
    VectorType totalType = VectorType::get(totalShape, elemTy);

    // Extract the last element along each scan tier via ExtractStridedSlice.
    SmallVector<int64_t> offsets(3 * rank, 0);
    offsets[d] = batchSize - 1;
    offsets[rank + d] = outerSize - 1;
    offsets[2 * rank + d] = elemSize - 1;
    SmallVector<int64_t> sizes(disShape);
    sizes[d] = 1;
    sizes[rank + d] = 1;
    sizes[2 * rank + d] = 1;
    SmallVector<int64_t> strides(3 * rank, 1);

    SmallVector<VectorValue> totals(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      totals[g] = vector::ExtractStridedSliceOp::create(
          rewriter, loc, inputs[g], offsets, sizes, strides);
    }

    // Flatten totals to scalars, do Hillis-Steele, unflatten.
    int64_t numTotalElements = totalType.getNumElements();
    VectorType flatType = VectorType::get({numTotalElements}, elemTy);

    SmallVector<VectorValue> flatTotals(numGroups);
    for (int64_t g = 0; g < numGroups; ++g) {
      flatTotals[g] =
          vector::ShapeCastOp::create(rewriter, loc, flatType, totals[g]);
    }

    // Hillis-Steele inclusive prefix scan on each scalar element.
    auto i32Ty = rewriter.getI32Type();
    SmallVector<VectorValue> inclusivePrefixes(numGroups);
    {
      SmallVector<VectorValue> current = flatTotals;

      for (int64_t s = 0; (1 << s) < threadTile; ++s) {
        int64_t offset = threadStride * (1 << s);
        Value offsetVal = arith::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(offset));
        Value widthVal = arith::ConstantOp::create(
            rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(width));

        // Initialize result vectors.
        SmallVector<VectorValue> next(numGroups);
        for (int64_t g = 0; g < numGroups; ++g) {
          next[g] = cast<VectorValue>(
              arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getZeroAttr(flatType))
                  .getResult());
        }

        for (int64_t i = 0; i < numTotalElements; ++i) {
          // Extract scalar from each group.
          SmallVector<Value> curScalars(numGroups);
          for (int64_t g = 0; g < numGroups; ++g) {
            curScalars[g] =
                vector::ExtractOp::create(rewriter, loc, current[g], i);
          }

          // Shuffle up each scalar.
          SmallVector<Value> shuffled(numGroups);
          Value valid;
          for (int64_t g = 0; g < numGroups; ++g) {
            auto shuffle =
                gpu::ShuffleOp::create(rewriter, loc, curScalars[g], offsetVal,
                                       widthVal, gpu::ShuffleMode::UP);
            shuffled[g] = shuffle.getShuffleResult();
            valid = shuffle.getValid();
          }

          // Combine: combined = combiner(shuffled, current)
          SmallVector<Value> combined =
              cloneCombinerScalar(rewriter, loc, body, shuffled, curScalars);

          // Select: if valid, use combined, else keep current.
          for (int64_t g = 0; g < numGroups; ++g) {
            Value selected = arith::SelectOp::create(
                rewriter, loc, valid, combined[g], curScalars[g]);
            next[g] =
                vector::InsertOp::create(rewriter, loc, selected, next[g], i);
          }
        }
        current = next;
      }
      inclusivePrefixes = current;
    }

    // Compute exclusive prefix: shuffle inclusive by threadStride.
    SmallVector<VectorValue> exclusivePrefixes(numGroups);
    SmallVector<Value> exclValidVec; // One valid flag per element.
    {
      Value strideVal = arith::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(threadStride));
      Value widthVal = arith::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(width));

      for (int64_t g = 0; g < numGroups; ++g) {
        exclusivePrefixes[g] =
            cast<VectorValue>(arith::ConstantOp::create(
                                  rewriter, loc, rewriter.getZeroAttr(flatType))
                                  .getResult());
      }

      Value exclValid;
      for (int64_t i = 0; i < numTotalElements; ++i) {
        SmallVector<Value> inclScalars(numGroups);
        for (int64_t g = 0; g < numGroups; ++g) {
          inclScalars[g] =
              vector::ExtractOp::create(rewriter, loc, inclusivePrefixes[g], i);
        }
        for (int64_t g = 0; g < numGroups; ++g) {
          auto shuffle =
              gpu::ShuffleOp::create(rewriter, loc, inclScalars[g], strideVal,
                                     widthVal, gpu::ShuffleMode::UP);
          exclusivePrefixes[g] = vector::InsertOp::create(
              rewriter, loc, shuffle.getShuffleResult(), exclusivePrefixes[g],
              i);
          exclValid = shuffle.getValid();
        }
      }

      // Reshape exclusive prefix back to totalShape.
      SmallVector<VectorValue> exclReshaped(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        exclReshaped[g] = vector::ShapeCastOp::create(rewriter, loc, totalType,
                                                      exclusivePrefixes[g]);
      }

      // Broadcast exclusive prefix to full distributed shape.
      SmallVector<VectorValue> exclBroadcast(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        exclBroadcast[g] = vector::BroadcastOp::create(rewriter, loc, disType,
                                                       exclReshaped[g]);
      }

      // Combine: combine(exclusive_broadcast, local_scan)
      SmallVector<Value> lhsVals(numGroups), rhsVals(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        lhsVals[g] = exclBroadcast[g];
        rhsVals[g] = inputs[g];
      }
      SmallVector<Value> combined = cloneCombinerVectorized(
          rewriter, loc, body, lhsVals, rhsVals, disType.getShape());

      // Select: if exclValid, use combined, else keep local_scan.
      Value exclValidSplat = vector::BroadcastOp::create(
          rewriter, loc,
          VectorType::get(disType.getShape(), rewriter.getI1Type()), exclValid);
      SmallVector<VectorValue> result(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        result[g] = cast<VectorValue>(
            arith::SelectOp::create(rewriter, loc, exclValidSplat, combined[g],
                                    inputs[g])
                .getResult());
      }
      return result;
    }
  }

  int64_t maxBitsPerShuffle;
};

} // namespace

void populateGPUDistributeNestedLayoutReductionPatterns(
    RewritePatternSet &patterns, int64_t subgroupSize,
    int64_t maxBitsPerShuffle) {
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
  patterns.add<DistributeAssociativeReduce>(patterns.getContext(), subgroupSize,
                                            maxBitsPerShuffle);
  patterns.add<DistributeAssociativeScan>(patterns.getContext(), subgroupSize,
                                          maxBitsPerShuffle);
}

} // namespace mlir::iree_compiler
