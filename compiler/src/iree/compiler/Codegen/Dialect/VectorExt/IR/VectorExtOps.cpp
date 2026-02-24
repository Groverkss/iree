// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

// Validate that the layout has the same shape as the input.
LogicalResult ToLayoutOp::verify() {
  return getLayout().isValidLayout(getInput().getType(), getLoc());
}

void ToLayoutOp::populateBoundsForShapedValueDim(
    Value value, int64_t dim, ValueBoundsConstraintSet &cstr) {
  cstr.bound(value)[dim] == cstr.getExpr(getInput(), dim);
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TransferGatherOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability TransferGatherOp::getSpeculatability() {
  if (isa<RankedTensorType>(getBase().getType())) {
    return Speculation::Speculatable;
  }
  return Speculation::NotSpeculatable;
}

void TransferGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<MemRefType>(getBase().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
  }
}

LogicalResult
mlir::iree_compiler::IREE::VectorExt::detail::verifyIndexedVectorOpInterface(
    Operation *operation) {
  auto op = cast<IndexedVectorOpInterface>(operation);
  VectorType vectorType = op.getVectorType();
  OperandRange indexVecs = op.getIndexVecs();
  Value mask = op.getMask();
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  // Check that we have the correct number of indexing maps.
  int64_t expectedNumIndexingMaps =
      /*baseIndexingMap=*/1 + /*indexVecIndexingMaps=*/indexVecs.size() +
      /*maskIndexingMap=*/(mask ? 1 : 0);
  if (expectedNumIndexingMaps != static_cast<int64_t>(indexingMaps.size())) {
    return op->emitOpError("expected ")
           << expectedNumIndexingMaps
           << " indexing maps, got: " << indexingMaps.size();
  }

  int64_t vectorRank = vectorType.getRank();
  int64_t indexSyms = indexVecs.size();
  for (AffineMap map : indexingMaps) {
    if (map.getNumDims() != vectorRank) {
      return op->emitOpError(
                 "expected all indexing maps to have number of dims "
                 "equal to vector rank. expected: ")
             << vectorRank << ", got: " << map.getNumDims() << " dims";
    }
    if (map.getNumSymbols() != indexSyms) {
      return op->emitOpError(
                 "expected all indexing maps to have number of symbols "
                 "equal to number of index vecs. expected: ")
             << indexSyms << ", got: " << map.getNumSymbols() << " syms";
    }
    for (AffineExpr expr : map.getResults()) {
      if (isa<AffineDimExpr, AffineSymbolExpr>(expr)) {
        continue;
      }
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        if (constExpr.getValue() != 0) {
          return op->emitOpError("expected constant 0 in indexing map, got: ")
                 << constExpr.getValue();
        }
        continue;
      }
      return op->emitOpError(
          "expected indexing map results to only be a dim, symbol, or 0");
    }
  }

  // Build the expected shape from a dim-only affine map by resolving each dim
  // expression against the vector shape. Returns failure if any non-dim
  // expression is found.
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  auto getExpectedShape =
      [&](AffineMap map) -> FailureOr<SmallVector<int64_t>> {
    SmallVector<int64_t> shape;
    for (AffineExpr expr : map.getResults()) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        shape.push_back(vectorShape[dim.getPosition()]);
      } else {
        return failure();
      }
    }
    return shape;
  };

  // Verify index vec shapes against their indexing maps.
  ArrayRef<AffineMap> vecMaps = ArrayRef(indexingMaps).slice(1, indexSyms);
  for (auto [i, map] : llvm::enumerate(vecMaps)) {
    FailureOr<SmallVector<int64_t>> expectedShape = getExpectedShape(map);
    if (failed(expectedShape)) {
      return op->emitOpError(
          "expected index vec indexing maps to only have dim exprs");
    }
    // Scalar index: map must have 0 results and type must be plain index.
    if (isa<IndexType>(indexVecs[i].getType())) {
      if (!expectedShape->empty()) {
        return op->emitOpError(
                   "expected empty indexing map for scalar index vec "
                   "at position ")
               << i;
      }
      continue;
    }
    ArrayRef<int64_t> actualShape =
        cast<VectorType>(indexVecs[i].getType()).getShape();
    if (ArrayRef<int64_t>(*expectedShape) != actualShape) {
      return op->emitOpError(
                 "mismatched vector shape for index vec at position ")
             << i << ". Expected: [" << *expectedShape << "]" << ", got: ["
             << actualShape << "]";
    }
  }

  // Verify mask shape against its indexing map.
  if (mask) {
    AffineMap maskMap = indexingMaps.back();
    FailureOr<SmallVector<int64_t>> expectedShape = getExpectedShape(maskMap);
    if (failed(expectedShape)) {
      return op->emitOpError(
          "expected mask indexing map to only have dim exprs");
    }
    ArrayRef<int64_t> actualShape = cast<VectorType>(mask.getType()).getShape();
    if (ArrayRef<int64_t>(*expectedShape) != actualShape) {
      return op->emitOpError("mismatched mask shape")
             << ". Expected: [" << *expectedShape << "]" << ", got: ["
             << actualShape << "]";
    }
  }

  return success();
}

// Fold and canonicalization helpers.

static int64_t getVectorRank(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getRank();
  }
  return 0;
}

struct IndexingMapFoldResult {
  Value operand;
  AffineMap indexingMap;
  bool changed;
};

using IndexingMapFolder = function_ref<IndexingMapFoldResult(
    int64_t index, Value val, AffineMap valMap, AffineMap &baseMap)>;

static Value foldTransferGatherIndexVecs(TransferGatherOp op,
                                         IndexingMapFolder valueFolder) {
  SmallVector<Value> indexedValues(op.getIndexVecs());
  SmallVector<AffineMap> indexingMaps(
      ArrayRef(op.getIndexingMapsArray()).slice(1, indexedValues.size()));

  AffineMap baseMap = op.getIndexingMapsArray().front();

  bool changed = false;
  SmallVector<Value> newIndexedValues;
  SmallVector<AffineMap> newIndexingMaps;
  llvm::DenseSet<int64_t> deletedSyms;
  for (auto [index, val, map] : llvm::enumerate(indexedValues, indexingMaps)) {
    auto [newVal, newMap, valChanged] = valueFolder(index, val, map, baseMap);
    changed |= valChanged;

    if (newVal) {
      newIndexedValues.push_back(newVal);
      newIndexingMaps.push_back(newMap);
    } else {
      deletedSyms.insert(index);
    }
  }

  // The mask is passed through the same folder as index vecs. Folders must
  // handle the mask case correctly — the mask's index is indexedValues.size()
  // which won't match any symbol in the base map, so index-based folds
  // (FoldSingleElementIndexVec, foldTransferGatherFromStep) will be no-ops
  // on the mask, while shape-based folds (broadcast, transpose) will apply.
  Value mask;
  AffineMap maskMap;
  if (op.getMask()) {
    auto [newMask, newMap, valChanged] =
        valueFolder(indexedValues.size(), op.getMask(),
                    op.getIndexingMapsArray().back(), baseMap);
    changed |= valChanged;
    if (newMask) {
      mask = newMask;
      maskMap = newMap;
    }
  }
  if (!changed) {
    return Value();
  }

  OpBuilder b(op);

  // Collect all the indexing maps.
  SmallVector<AffineMap> updatedIndexingMaps;
  updatedIndexingMaps.push_back(baseMap);
  updatedIndexingMaps.append(newIndexingMaps);
  if (op.getMask()) {
    updatedIndexingMaps.push_back(maskMap);
  }

  // Delete the deleted symbols from these maps.
  if (!deletedSyms.empty()) {
    SmallVector<AffineExpr> symReplacements;
    int currSym = 0;
    for (auto i : llvm::seq<int>(baseMap.getNumSymbols())) {
      if (deletedSyms.contains(i)) {
        symReplacements.push_back(b.getAffineConstantExpr(0));
      } else {
        symReplacements.push_back(b.getAffineSymbolExpr(currSym));
        ++currSym;
      }
    }
    for (AffineMap &map : updatedIndexingMaps) {
      map = map.replaceDimsAndSymbols({}, symReplacements, map.getNumDims(),
                                      currSym);
    }
  }

  SmallVector<Value> operands;
  operands.push_back(op.getBase());
  llvm::append_range(operands, op.getOffsets());
  llvm::append_range(operands, newIndexedValues);
  operands.push_back(op.getPadding());
  if (mask) {
    operands.push_back(mask);
  }

  op.setIndexingMapsAttr(b.getAffineMapArrayAttr(updatedIndexingMaps));
  op->setOperands(operands);
  op.getProperties().setOperandSegmentSizes(
      {1, static_cast<int32_t>(op.getOffsets().size()),
       static_cast<int32_t>(newIndexedValues.size()), 1,
       static_cast<int32_t>(mask ? 1 : 0)});

  return op.getResult();
}

static Value foldTransferGatherFromBroadcast(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t, Value operand, AffineMap map,
         AffineMap &) -> IndexingMapFoldResult {
        auto broadcast = operand.getDefiningOp<vector::BroadcastOp>();
        if (!broadcast) {
          return {operand, map, false};
        }

        int64_t sourceRank = getVectorRank(broadcast.getSourceType());
        int64_t operandRank = getVectorRank(broadcast.getResultVectorType());
        AffineMap newMap =
            map.getSliceMap(operandRank - sourceRank, sourceRank);
        return {broadcast.getSource(), newMap, true};
      });
}

static Value foldTransferGatherFromTranspose(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t, Value operand, AffineMap map,
         AffineMap &) -> IndexingMapFoldResult {
        auto transpose = operand.getDefiningOp<vector::TransposeOp>();
        if (!transpose) {
          return {operand, map, false};
        }

        AffineMap newMap =
            AffineMap::getPermutationMap(
                invertPermutationVector(transpose.getPermutation()),
                transpose.getContext())
                .compose(map);
        return {transpose.getVector(), newMap, true};
      });
}

static Value foldTransferGatherFromStep(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t index, Value operand, AffineMap map,
         AffineMap &baseMap) -> IndexingMapFoldResult {
        auto step = operand.getDefiningOp<vector::StepOp>();
        if (!step) {
          return {operand, map, false};
        }

        assert(map.getNumResults() == 1);
        // Replace the symbol in the base map with the dim expression from the
        // index vec map, making this dimension contiguous.
        SmallVector<AffineExpr> newResults;
        for (AffineExpr expr : baseMap.getResults()) {
          if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
            if (sym.getPosition() == index) {
              expr = map.getResult(0);
            }
          }
          newResults.push_back(expr);
        }
        baseMap = AffineMap::get(baseMap.getNumDims(), baseMap.getNumSymbols(),
                                 newResults, baseMap.getContext());
        return {Value(), AffineMap(), true};
      });
}

OpFoldResult TransferGatherOp::fold(FoldAdaptor adaptor) {
  // Fold all-true splat mask by dropping the mask operand. Since every
  // position is unmasked, the mask indexing map is irrelevant. This runs
  // before the index vec folds below; the ordering does not matter because
  // the index vec folds operate independently of the mask.
  if (auto maskAttr =
          dyn_cast_if_present<DenseElementsAttr>(adaptor.getMask())) {
    if (maskAttr.isSplat() && maskAttr.getSplatValue<bool>()) {
      int32_t numOffsets = static_cast<int32_t>(getOffsets().size());
      int32_t numIndexVecs = static_cast<int32_t>(getIndexVecs().size());

      Builder b(getContext());
      SmallVector<AffineMap> maps = getIndexingMapsArray();
      maps.pop_back();
      setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));

      SmallVector<Value> operands;
      operands.push_back(getBase());
      llvm::append_range(operands, getOffsets());
      llvm::append_range(operands, getIndexVecs());
      operands.push_back(getPadding());

      getProperties().setOperandSegmentSizes(
          {1, numOffsets, numIndexVecs, 1, 0});
      (*this)->setOperands(operands);

      return getResult();
    }
  }
  if (auto res = foldTransferGatherFromBroadcast(*this)) {
    return res;
  }
  if (auto res = foldTransferGatherFromTranspose(*this)) {
    return res;
  }
  if (auto res = foldTransferGatherFromStep(*this)) {
    return res;
  }
  return OpFoldResult();
}

/// Apply an affine map transformation to a vector using broadcast and
/// transpose operations.
static Value applyTransformMapToVector(PatternRewriter &rewriter, Location loc,
                                       Value source, AffineMap map,
                                       ArrayRef<int64_t> targetShape) {
  auto sourceType = cast<VectorType>(source.getType());
  int64_t targetRank = map.getNumDims();
  int64_t sourceRank = sourceType.getRank();

  assert(map.getNumResults() == sourceRank &&
         "Map results must match source rank");

  // If already the right shape, no transformation needed.
  if (sourceRank == targetRank) {
    bool isIdentity = true;
    for (unsigned i = 0; i < sourceRank; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
      if (!dimExpr || dimExpr.getPosition() != i) {
        isIdentity = false;
        break;
      }
    }
    if (isIdentity) {
      return source;
    }
  }

  // Build direct mapping: for each target dim, which source dim provides it
  SmallVector<int64_t> targetDimToSourceDim(targetRank, -1);
  for (int64_t srcDim = 0; srcDim < sourceRank; ++srcDim) {
    AffineExpr expr = map.getResult(srcDim);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      targetDimToSourceDim[dimExpr.getPosition()] = srcDim;
    }
  }

  int64_t numBroadcastDims = llvm::count(targetDimToSourceDim, -1);

  // Build broadcast shape: [broadcast dim sizes..., source shape...]
  SmallVector<int64_t> broadcastShape;
  for (int64_t i = 0; i < targetRank; ++i) {
    if (targetDimToSourceDim[i] == -1) {
      broadcastShape.push_back(targetShape[i]);
    }
  }
  for (int64_t i = 0; i < sourceRank; ++i) {
    broadcastShape.push_back(sourceType.getDimSize(i));
  }

  // Broadcast to add dimensions
  VectorType broadcastType =
      VectorType::get(broadcastShape, sourceType.getElementType());
  Value result = source;
  if (broadcastType != sourceType) {
    result = vector::BroadcastOp::create(rewriter, loc, broadcastType, source);
  }

  // Compute transpose permutation
  SmallVector<int64_t> transposePerm(targetRank);
  int64_t bcastIdx = 0;
  for (int64_t i = 0; i < targetRank; ++i) {
    transposePerm[i] = targetDimToSourceDim[i] == -1
                           ? bcastIdx++
                           : numBroadcastDims + targetDimToSourceDim[i];
  }

  if (!llvm::equal(transposePerm, llvm::seq<int64_t>(0, targetRank))) {
    result = vector::TransposeOp::create(rewriter, loc, result, transposePerm);
  }

  return result;
}

struct FoldSingleElementIndexVec final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](int64_t index, Value indexVec, AffineMap map,
                              AffineMap &baseMap) -> IndexingMapFoldResult {
      bool isScalar = isa<IndexType>(indexVec.getType());
      if (!isScalar) {
        auto vectorTy = cast<VectorType>(indexVec.getType());
        if (vectorTy.getNumElements() != 1) {
          return {indexVec, map, false};
        }
      }

      // Find which source dim this symbol corresponds to.
      AffineExpr symbolExpr = getAffineSymbolExpr(index, op.getContext());
      int64_t sourceDim = -1;
      for (auto [i, expr] : llvm::enumerate(baseMap.getResults())) {
        if (expr == symbolExpr) {
          sourceDim = i;
          break;
        }
      }
      if (sourceDim < 0) {
        return {indexVec, map, false};
      }

      // Extract the scalar and add it to the corresponding base offset.
      OpOperand &baseOffset = op.getOffsetsMutable()[sourceDim];
      Value extracted = indexVec;
      if (!isScalar) {
        auto vectorTy = cast<VectorType>(indexVec.getType());
        extracted = vector::ExtractOp::create(
            rewriter, op.getLoc(), indexVec,
            SmallVector<int64_t>(vectorTy.getRank(), 0));
      }

      AffineExpr d0, d1;
      bindDims(op.getContext(), d0, d1);

      Value newIndex = affine::makeComposedAffineApply(
                           rewriter, op.getLoc(), d0 + d1,
                           ArrayRef<OpFoldResult>{baseOffset.get(), extracted})
                           .getResult();
      baseOffset.set(newIndex);

      return {Value(), AffineMap(), true};
    };

    Value newVal = foldTransferGatherIndexVecs(op, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

/// Fold `arith.addi(something, broadcast(scalar))` index vecs by absorbing
/// the scalar into the base offset. This handles the common pattern after
/// unrolling where offsets get added to index vectors as broadcasts.
struct FoldIndexVecAddBroadcast final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](int64_t index, Value indexVec, AffineMap map,
                              AffineMap &baseMap) -> IndexingMapFoldResult {
      auto addOp = indexVec.getDefiningOp<arith::AddIOp>();
      if (!addOp) {
        return {indexVec, map, false};
      }

      // Try both operand orders (addi is commutative).
      Value scalarSrc;
      Value remaining;
      for (auto [lhs, rhs] : {std::pair(addOp.getLhs(), addOp.getRhs()),
                              std::pair(addOp.getRhs(), addOp.getLhs())}) {
        auto broadcast = lhs.getDefiningOp<vector::BroadcastOp>();
        if (broadcast && isa<IndexType>(broadcast.getSourceType())) {
          scalarSrc = broadcast.getSource();
          remaining = rhs;
          break;
        }
      }
      if (!scalarSrc) {
        return {indexVec, map, false};
      }

      // Find which source dim this symbol corresponds to.
      AffineExpr symbolExpr = getAffineSymbolExpr(index, op.getContext());
      int64_t sourceDim = -1;
      for (auto [i, expr] : llvm::enumerate(baseMap.getResults())) {
        if (expr == symbolExpr) {
          sourceDim = i;
          break;
        }
      }
      if (sourceDim < 0) {
        return {indexVec, map, false};
      }

      // Add the scalar to the corresponding base offset.
      OpOperand &baseOffset = op.getOffsetsMutable()[sourceDim];

      AffineExpr d0, d1;
      bindDims(op.getContext(), d0, d1);

      Value newOffset = affine::makeComposedAffineApply(
                            rewriter, op.getLoc(), d0 + d1,
                            ArrayRef<OpFoldResult>{baseOffset.get(), scalarSrc})
                            .getResult();
      baseOffset.set(newOffset);

      // Replace index vec with the non-broadcast addend.
      return {remaining, map, true};
    };

    Value newVal = foldTransferGatherIndexVecs(op, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

/// Replace an all-false masked transfer_gather with a broadcast of the padding.
/// Only handles splat constants; non-splat all-false constants are expected to
/// be canonicalized to splat form beforehand.
struct FoldAllFalseMaskTransferGather final
    : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    Value mask = op.getMask();
    if (!mask) {
      return rewriter.notifyMatchFailure(op, "no mask operand");
    }

    DenseElementsAttr maskAttr;
    if (!matchPattern(mask, m_Constant(&maskAttr))) {
      return rewriter.notifyMatchFailure(op, "mask is not a constant");
    }

    if (!maskAttr.isSplat() || maskAttr.getSplatValue<bool>()) {
      return rewriter.notifyMatchFailure(op, "mask is not splat false");
    }

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getType(),
                                                     op.getPadding());
    return success();
  }
};

struct FoldContiguousGatherToTransferRead final
    : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getIndexVecs().empty()) {
      return failure();
    }

    AffineMap permutationMap = op.getBasePermutationMap();

    Value mask = op.getMask();
    if (mask) {
      // First, apply the mask indexing map if it's not identity.
      AffineMap maskMap = op.getIndexingMapsArray().back();
      ArrayRef<int64_t> targetShape = op.getType().getShape();
      if (!maskMap.isIdentity()) {
        mask = applyTransformMapToVector(rewriter, op.getLoc(), mask, maskMap,
                                         targetShape);
      }
      // Then, compress the mask to match transfer_read's expected mask type
      // (which drops broadcast dims from the permutation map).
      auto expectedMaskType =
          vector::inferTransferOpMaskType(op.getType(), permutationMap);
      if (mask.getType() != expectedMaskType) {
        mask = vector::ShapeCastOp::create(rewriter, op.getLoc(),
                                           expectedMaskType, mask);
      }
    }

    SmallVector<bool> inBoundsVec(op.getType().getRank(), true);
    ArrayAttr inBounds = rewriter.getBoolArrayAttr(inBoundsVec);

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), op.getBase(), op.getOffsets(), permutationMap,
        op.getPadding(), mask, inBounds);
    return success();
  };
};

void TransferGatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *ctx) {
  results
      .add<FoldSingleElementIndexVec, FoldIndexVecAddBroadcast,
           FoldAllFalseMaskTransferGather, FoldContiguousGatherToTransferRead>(
          ctx);
}

//===----------------------------------------------------------------------===//
// TransferScatterOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability TransferScatterOp::getSpeculatability() {
  if (isa<RankedTensorType>(getBase().getType())) {
    return Speculation::Speculatable;
  }
  return Speculation::NotSpeculatable;
}

void TransferScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<MemRefType>(getBase().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
  }
}

LogicalResult TransferScatterOp::verify() {
  // Scatter-specific checks.
  if (hasTensorSemantics()) {
    if (!getResult()) {
      return emitOpError("expected result for tensor operand");
    }
    if (getResult().getType() != getBase().getType()) {
      return emitOpError("result type must match base type");
    }
  } else {
    // Memref semantics: no result expected.
    if (getResult()) {
      return emitOpError("unexpected result for memref operand");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyYieldForArgCompare(YieldOp yieldOp,
                                              ArgCompareOp argCompareOp) {
  unsigned numOperands = yieldOp.getNumOperands();
  if (numOperands != 1) {
    return yieldOp.emitOpError("expected 1 yield operand, but got ")
           << numOperands;
  }

  Type yieldType = yieldOp.getOperand(0).getType();
  if (!yieldType.isInteger(1)) {
    return yieldOp.emitOpError(
               "expected yield operand to have type i1, but got ")
           << yieldType;
  }

  return success();
}

static LogicalResult
verifyYieldForAssociativeOp(YieldOp yieldOp,
                            ArrayRef<Type> expectedElementTypes) {
  int64_t numOperands = yieldOp.getNumOperands();
  int64_t expected = expectedElementTypes.size();
  if (numOperands != expected) {
    return yieldOp.emitOpError("expected ")
           << expected << " yield operand(s), but got " << numOperands;
  }
  for (int64_t i = 0, e = expected; i < e; ++i) {
    Type actual = yieldOp.getOperand(i).getType();
    if (actual != expectedElementTypes[i]) {
      return yieldOp.emitOpError("expected yield operand #")
             << i << " to have type " << expectedElementTypes[i] << ", but got "
             << actual;
    }
  }
  return success();
}

LogicalResult YieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  if (auto argCompareOp = dyn_cast<ArgCompareOp>(parent)) {
    return verifyYieldForArgCompare(*this, argCompareOp);
  }

  // Both AssociativeReduceOp and AssociativeScanOp share the same yield
  // verification: each yield operand must match the element type of the
  // corresponding input.
  auto getElemTypes = [](OperandRange inputs) {
    SmallVector<Type> elemTypes;
    for (Value input : inputs) {
      elemTypes.push_back(cast<VectorType>(input.getType()).getElementType());
    }
    return elemTypes;
  };
  if (auto reduceOp = dyn_cast<AssociativeReduceOp>(parent)) {
    return verifyYieldForAssociativeOp(*this,
                                       getElemTypes(reduceOp.getInputs()));
  }
  if (auto scanOp = dyn_cast<AssociativeScanOp>(parent)) {
    return verifyYieldForAssociativeOp(*this, getElemTypes(scanOp.getInputs()));
  }

  assert(false && "ParentOneOf trait should have rejected this");
  return failure();
}

//===----------------------------------------------------------------------===//
// ArgCompareOp
//===----------------------------------------------------------------------===//

LogicalResult ArgCompareOp::verify() {
  Operation *op = getOperation();

  VectorType inputType = getInputValueType();
  VectorType initValueType = getInitValueType();
  VectorType initIndexType = getInitIndexType();

  int64_t inputRank = inputType.getRank();
  int64_t initValueRank = initValueType.getRank();
  int64_t initIndexRank = initIndexType.getRank();
  int64_t dimension = getDimension();

  if (dimension < 0 || dimension >= inputRank) {
    return op->emitOpError("dimension ")
           << dimension << " is out of range [0, " << inputRank << ")";
  }

  if (initValueRank != inputRank - 1) {
    return op->emitOpError("init value rank (")
           << initValueRank << ") must be input rank - 1 (" << (inputRank - 1)
           << ")";
  }

  if (initIndexRank != inputRank - 1) {
    return op->emitOpError("init index rank (")
           << initIndexRank << ") must be input rank - 1 (" << (inputRank - 1)
           << ")";
  }

  SmallVector<int64_t> expectedShape;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (i != dimension) {
      expectedShape.push_back(inputType.getDimSize(i));
    }
  }

  ArrayRef<int64_t> initValueShape = initValueType.getShape();
  if (expectedShape != initValueShape) {
    return op->emitOpError(
               "init value shape must match input shape with reduction "
               "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: " << llvm::interleaved_array(initValueShape);
  }

  ArrayRef<int64_t> initIndexShape = initIndexType.getShape();
  if (expectedShape != initIndexShape) {
    return op->emitOpError(
               "init index shape must match input shape with reduction "
               "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: " << llvm::interleaved_array(initIndexShape);
  }

  Type initIndexElementType = initIndexType.getElementType();
  if (!isa<IntegerType, IndexType>(initIndexElementType)) {
    return op->emitOpError(
               "init index must have integer or index element type, but got ")
           << initIndexElementType;
  }

  if (hasExplicitIndexInput()) {
    VectorType inputIndexType = getInputIndexType();
    ArrayRef<int64_t> inputIndexShape = inputIndexType.getShape();
    ArrayRef<int64_t> inputValueShape = inputType.getShape();

    if (inputIndexShape != inputValueShape) {
      return op->emitOpError(
                 "explicit-index mode: value and index inputs must have the "
                 "same shape. ")
             << "Value shape: " << llvm::interleaved_array(inputValueShape)
             << ", index shape: " << llvm::interleaved_array(inputIndexShape);
    }

    Type inputIndexElementType = getInputIndexElementType();
    Type initIndexElementType = initIndexType.getElementType();

    if (!isa<IntegerType, IndexType>(inputIndexElementType)) {
      return op->emitOpError("explicit-index mode: index input must have "
                             "integer or index element type, but got ")
             << inputIndexElementType;
    }

    if (inputIndexElementType != initIndexElementType) {
      return op->emitOpError(
                 "explicit-index mode: input and init index element types "
                 "must match. ")
             << "Input index type: " << inputIndexElementType
             << ", init index type: " << initIndexElementType;
    }

    if (getIndexBase()) {
      return op->emitOpError(
          "index_base must not be used with explicit indices");
    }
  }

  // Region structure is enforced by ODS (SizedRegion<1> and
  // SingleBlockImplicitTerminator), so we can directly access it.
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("comparator region must have exactly 2 arguments");
  }

  Type inputElementType = inputType.getElementType();
  Type arg0Type = block.getArgument(0).getType();
  Type arg1Type = block.getArgument(1).getType();

  if (arg0Type != inputElementType || arg1Type != inputElementType) {
    return op->emitOpError(
               "comparator arguments must match input value element type. ")
           << "Expected: " << inputElementType << ", but got: " << arg0Type
           << " and " << arg1Type;
  }

  // Since ArgCompareOp is marked Pure, all operations in the comparator must
  // also be pure.
  for (Operation &op : block.getOperations()) {
    if (!isPure(&op)) {
      return op.emitOpError(
          "comparator region must contain only pure operations");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shared verifiers for associative reduce/scan ops.
//===----------------------------------------------------------------------===//

static LogicalResult verifyReductionDims(Operation *op, int64_t inputRank,
                                         ArrayRef<int64_t> reductionDims) {
  for (int64_t i = 0, e = reductionDims.size(); i < e; ++i) {
    int64_t dim = reductionDims[i];
    if (dim < 0 || dim >= inputRank) {
      return op->emitOpError("reduction dimension ")
             << dim << " is out of range [0, " << inputRank << ")";
    }
    if (i > 0 && reductionDims[i] <= reductionDims[i - 1]) {
      return op->emitOpError(
                 "reduction dimensions must be sorted and unique, but got ")
             << llvm::interleaved_array(reductionDims);
    }
  }
  return success();
}

static LogicalResult verifyAssociativeCombiner(Operation *op,
                                               OperandRange inputs,
                                               Region &combiner) {
  int64_t numGroups = inputs.size();
  if (numGroups < 1) {
    return op->emitOpError("expected at least one input");
  }

  // All inputs must have the same shape.
  auto firstType = cast<VectorType>(inputs.front().getType());
  for (Value input : inputs.drop_front()) {
    auto inputType = cast<VectorType>(input.getType());
    if (inputType.getShape() != firstType.getShape()) {
      return op->emitOpError("all inputs must have the same shape, but got ")
             << firstType << " and " << inputType;
    }
  }

  // Combiner region: 2*N block args.
  Block &block = combiner.front();
  int64_t expectedArgs = 2 * numGroups;
  if (static_cast<int64_t>(block.getNumArguments()) != expectedArgs) {
    return op->emitOpError("combiner region must have exactly ")
           << expectedArgs << " arguments (2 * " << numGroups
           << " operand groups), but got " << block.getNumArguments();
  }

  // Block arg types: [elem0, elem1, ..., elem0, elem1, ...]
  for (int64_t i = 0; i < expectedArgs; ++i) {
    int64_t groupIdx = i % numGroups;
    Type expectedTy =
        cast<VectorType>(inputs[groupIdx].getType()).getElementType();
    Type actualTy = block.getArgument(i).getType();
    if (actualTy != expectedTy) {
      return op->emitOpError("combiner argument #")
             << i << " must have type " << expectedTy << ", but got "
             << actualTy;
    }
  }

  // All ops must be Pure.
  for (Operation &bodyOp : block.getOperations()) {
    if (!isPure(&bodyOp)) {
      return bodyOp.emitOpError(
          "combiner region must contain only pure operations");
    }
  }

  return success();
}

/// Compute the expected shape after removing reduction dimensions.
static SmallVector<int64_t> getReducedShape(VectorType inputType,
                                            ArrayRef<int64_t> reductionDims) {
  SmallVector<int64_t> shape;
  for (int64_t i = 0, e = inputType.getRank(); i < e; ++i) {
    if (!llvm::is_contained(reductionDims, i)) {
      shape.push_back(inputType.getDimSize(i));
    }
  }
  return shape;
}

//===----------------------------------------------------------------------===//
// AssociativeReduceOp
//===----------------------------------------------------------------------===//

LogicalResult AssociativeReduceOp::verify() {
  OperandRange inputs = getInputs();
  ResultRange results = getResults();

  if (inputs.size() != results.size()) {
    return emitOpError("expected same number of inputs and results, but got ")
           << inputs.size() << " inputs and " << results.size() << " results";
  }

  auto inputType = cast<VectorType>(inputs.front().getType());
  if (failed(verifyReductionDims(getOperation(), inputType.getRank(),
                                 getReductionDims()))) {
    return failure();
  }

  if (failed(
          verifyAssociativeCombiner(getOperation(), inputs, getCombiner()))) {
    return failure();
  }

  // Verify result shapes and element types.
  SmallVector<int64_t> expectedResultShape =
      getReducedShape(inputType, getReductionDims());
  for (auto [i, result, input] : llvm::enumerate(results, inputs)) {
    auto resultType = cast<VectorType>(result.getType());
    auto inType = cast<VectorType>(input.getType());

    if (ArrayRef<int64_t>(expectedResultShape) != resultType.getShape()) {
      return emitOpError("result #")
             << i
             << " shape must match input shape with reduction dimensions "
                "removed. Expected: "
             << llvm::interleaved_array(expectedResultShape)
             << ", but got: " << llvm::interleaved_array(resultType.getShape());
    }

    if (resultType.getElementType() != inType.getElementType()) {
      return emitOpError("result #")
             << i << " element type must match input #" << i
             << " element type. Expected: " << inType.getElementType()
             << ", but got: " << resultType.getElementType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AssociativeScanOp
//===----------------------------------------------------------------------===//

LogicalResult AssociativeScanOp::verify() {
  OperandRange inputs = getInputs();
  ResultRange results = getResults();

  if (inputs.size() != results.size()) {
    return emitOpError("expected same number of inputs and results, but got ")
           << inputs.size() << " inputs and " << results.size() << " results";
  }

  auto inputType = cast<VectorType>(inputs.front().getType());
  if (failed(verifyReductionDims(getOperation(), inputType.getRank(),
                                 getReductionDims()))) {
    return failure();
  }

  if (failed(
          verifyAssociativeCombiner(getOperation(), inputs, getCombiner()))) {
    return failure();
  }

  // Results must match input types.
  for (auto [i, input, result] : llvm::enumerate(inputs, results)) {
    if (input.getType() != result.getType()) {
      return emitOpError("result #") << i << " type must match input #" << i
                                     << " type. Expected: " << input.getType()
                                     << ", but got: " << result.getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AssociativeReduceOp canonicalization
//===----------------------------------------------------------------------===//

/// Fold zero-dim reduce to identity: when reduction_dims is empty, the result
/// equals the input.
struct FoldZeroDimReduce final : OpRewritePattern<AssociativeReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssociativeReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getReductionDims().empty()) {
      return failure();
    }
    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

/// Maximum size of a reduction dimension that will be inlined by the
/// InlineSmallReduce/InlineSmallScan canonicalization. Keeps code size bounded.
/// In practice, the distribution combine step stacks into dims of size 2, so
/// this primarily handles small dimensions (2-8) produced during lowering.
static constexpr int64_t kMaxInlineReductionSize = 8;

/// Extract a slice at position `idx` along dimension `dim`, removing that
/// dimension from the result shape via shape_cast.
static Value extractSliceAlongDim(PatternRewriter &rewriter, Location loc,
                                  Value source, int64_t dim, int64_t idx) {
  auto sourceType = cast<VectorType>(source.getType());
  int64_t rank = sourceType.getRank();

  // Build offsets/sizes/strides for extract_strided_slice.
  SmallVector<int64_t> offsets(rank, 0);
  SmallVector<int64_t> sizes(sourceType.getShape());
  SmallVector<int64_t> strides(rank, 1);
  offsets[dim] = idx;
  sizes[dim] = 1;

  Value slice = vector::ExtractStridedSliceOp::create(rewriter, loc, source,
                                                      offsets, sizes, strides);

  // Shape cast to remove the size-1 dimension.
  SmallVector<int64_t> resultShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      resultShape.push_back(sourceType.getDimSize(i));
    }
  }
  VectorType resultType =
      VectorType::get(resultShape, sourceType.getElementType());
  return vector::ShapeCastOp::create(rewriter, loc, resultType, slice);
}

/// Apply the combiner body to two groups of vector operands (lhs and rhs),
/// cloning each body op with vector-typed results.
/// Assumption: all combiner body ops are element-wise (e.g., arith ops) and
/// can be vectorized by widening result types.
/// TODO: Unify with cloneCombinerVectorized in
/// GPUNestedLayoutDistributionReductionPatterns.cpp.
static SmallVector<Value>
applyCombinerToVectors(PatternRewriter &rewriter, Location loc, Block &body,
                       ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                       ArrayRef<int64_t> resultShape) {
  int64_t numGroups = lhs.size();
  IRMapping mapping;
  for (int64_t g = 0; g < numGroups; ++g) {
    mapping.map(body.getArgument(g), lhs[g]);
    mapping.map(body.getArgument(numGroups + g), rhs[g]);
  }

  for (Operation &bodyOp : body.without_terminator()) {
    Operation *cloned = rewriter.clone(bodyOp, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(bodyOp.getResults(), cloned->getResults())) {
      VectorType resultVecTy =
          VectorType::get(resultShape, oldResult.getType());
      newResult.setType(resultVecTy);
      mapping.map(oldResult, newResult);
    }
  }

  auto yieldOp = cast<YieldOp>(body.getTerminator());
  SmallVector<Value> results;
  for (int64_t g = 0; g < numGroups; ++g) {
    results.push_back(mapping.lookupOrDefault(yieldOp.getOperand(g)));
  }
  return results;
}

/// Inline a small associative_reduce by extracting slices along the reduction
/// dims and folding them with cloned combiner ops. This is the key
/// canonicalization for the distribution combine step, where values are stacked
/// into a dim of size 2 and then reduced.
struct InlineSmallReduce final : OpRewritePattern<AssociativeReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssociativeReduceOp op,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> reductionDims = op.getReductionDims();
    if (reductionDims.empty()) {
      return failure();
    }

    VectorType inputType = op.getInputType();
    for (int64_t d : reductionDims) {
      int64_t size = inputType.getDimSize(d);
      if (ShapedType::isDynamic(size) || size > kMaxInlineReductionSize) {
        return failure();
      }
    }

    Location loc = op.getLoc();
    int64_t numGroups = op.getNumOperandGroups();
    Block &body = op.getCombiner().front();

    // Reduce one dim at a time, from highest to lowest (to keep indices
    // stable).
    SmallVector<Value> current(op.getInputs());
    for (int64_t ri = reductionDims.size() - 1; ri >= 0; --ri) {
      int64_t dim = reductionDims[ri];
      // Adjust for dims already removed above this one.
      int64_t adjustedDim = dim;
      for (int64_t rj = ri + 1; rj < static_cast<int64_t>(reductionDims.size());
           ++rj) {
        if (reductionDims[rj] > dim) {
          break;
        }
        --adjustedDim;
      }

      auto curType = cast<VectorType>(current[0].getType());
      int64_t dimSize = curType.getDimSize(adjustedDim);

      // Extract slice at index 0 as initial accumulator.
      SmallVector<Value> acc(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        acc[g] =
            extractSliceAlongDim(rewriter, loc, current[g], adjustedDim, 0);
      }

      // Compute result shape after removing this dim.
      SmallVector<int64_t> resultShape;
      for (int64_t i = 0, e = curType.getRank(); i < e; ++i) {
        if (i != adjustedDim) {
          resultShape.push_back(curType.getDimSize(i));
        }
      }

      // Left-fold remaining slices.
      for (int64_t i = 1; i < dimSize; ++i) {
        SmallVector<Value> slices(numGroups);
        for (int64_t g = 0; g < numGroups; ++g) {
          slices[g] =
              extractSliceAlongDim(rewriter, loc, current[g], adjustedDim, i);
        }
        acc = applyCombinerToVectors(rewriter, loc, body, acc, slices,
                                     resultShape);
      }

      current = acc;
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

void AssociativeReduceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldZeroDimReduce, InlineSmallReduce>(context);
}

//===----------------------------------------------------------------------===//
// AssociativeScanOp canonicalization
//===----------------------------------------------------------------------===//

/// Fold zero-dim scan to identity: when reduction_dims is empty, the result
/// equals the input.
struct FoldZeroDimScan final : OpRewritePattern<AssociativeScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssociativeScanOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getReductionDims().empty()) {
      return failure();
    }
    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

/// Inline a small associative_scan by extracting slices along scan dims
/// and accumulating with cloned combiner ops. Unlike InlineSmallReduce,
/// this preserves dimensions (prefix scan semantics).
struct InlineSmallScan final : OpRewritePattern<AssociativeScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssociativeScanOp op,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> scanDims = op.getReductionDims();
    if (scanDims.empty()) {
      return failure();
    }

    VectorType inputType = op.getInputType();
    for (int64_t d : scanDims) {
      int64_t size = inputType.getDimSize(d);
      if (ShapedType::isDynamic(size) || size > kMaxInlineReductionSize) {
        return failure();
      }
    }

    Location loc = op.getLoc();
    int64_t numGroups = op.getNumOperandGroups();
    Block &body = op.getCombiner().front();

    // Process one scan dim at a time.
    SmallVector<Value> current(op.getInputs());
    for (int64_t d : scanDims) {
      auto curType = cast<VectorType>(current[0].getType());
      int64_t dimSize = curType.getDimSize(d);

      // Result shape for extracted slices (dim d becomes size 1, not removed).
      SmallVector<int64_t> sliceResultShape(curType.getShape());
      sliceResultShape[d] = 1;

      // Extract slice at index 0 as initial accumulator.
      SmallVector<Value> acc(numGroups);
      for (int64_t g = 0; g < numGroups; ++g) {
        acc[g] = extractSliceAlongDim(rewriter, loc, current[g], d, 0);
      }

      // Compute result shape for combiner (rank-1, dim d removed).
      SmallVector<int64_t> combinerShape;
      for (int64_t i = 0, e = curType.getRank(); i < e; ++i) {
        if (i != d) {
          combinerShape.push_back(curType.getDimSize(i));
        }
      }

      // Build result by inserting accumulated slices back.
      SmallVector<Value> result(current);

      // Slice 0 stays as-is. Left-fold remaining slices.
      for (int64_t i = 1; i < dimSize; ++i) {
        SmallVector<Value> slices(numGroups);
        for (int64_t g = 0; g < numGroups; ++g) {
          slices[g] = extractSliceAlongDim(rewriter, loc, current[g], d, i);
        }
        acc = applyCombinerToVectors(rewriter, loc, body, acc, slices,
                                     combinerShape);
        // Insert acc back into result at position i along dim d.
        for (int64_t g = 0; g < numGroups; ++g) {
          Value accReshaped = vector::ShapeCastOp::create(
              rewriter, loc,
              VectorType::get(sliceResultShape, curType.getElementType()),
              acc[g]);
          SmallVector<int64_t> offsets(curType.getRank(), 0);
          offsets[d] = i;
          SmallVector<int64_t> strides(curType.getRank(), 1);
          result[g] = vector::InsertStridedSliceOp::create(
              rewriter, loc, accReshaped, result[g], offsets, strides);
        }
      }

      current = result;
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

void AssociativeScanOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<FoldZeroDimScan, InlineSmallScan>(context);
}

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
