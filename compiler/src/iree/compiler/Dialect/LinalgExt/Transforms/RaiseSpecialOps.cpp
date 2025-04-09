// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_RAISESPECIALOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {
/// Matches a linalg.generic operation reading data from a tensor `source`
/// using tensor.extract, and raises the `source` tensor to an input of the
/// linalg operation.
static FailureOr<linalg::GenericOp>
raiseTensorExtractToInput(linalg::GenericOp linalgOp, RewriterBase &rewriter) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }
  if (!isElementwise(linalgOp)) {
    return failure();
  }
  if (!llvm::hasSingleElement(linalgOp.getResults())) {
    return failure();
  }

  // Find a tensor.extract op in the linalgOp body.
  auto extractOps = linalgOp.getBody()->getOps<tensor::ExtractOp>();
  if (!llvm::hasSingleElement(extractOps)) {
    return failure();
  }
  tensor::ExtractOp extractOp = *extractOps.begin();
  auto resultType = dyn_cast<TensorType>(linalgOp.getResult(0).getType());
  if (!resultType) {
    return failure();
  }

  TensorType tensorType = extractOp.getTensor().getType();
  ArrayRef<int64_t> sourceShape = tensorType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  SmallVector<AffineExpr> exprs;
  for (auto [idx, indexValue] : llvm::enumerate(extractOp.getIndices())) {
    // For raising, the indexing value must be one of the following:
    //    1. A constant value.
    //    2. A value defined outside the body.
    //    3. A linalg.index.

    // 1. Indexing value is a constant.
    APInt constantIndex;
    if (matchPattern(indexValue, m_ConstantInt(&constantIndex))) {
      // Restrict to cases where the constant is 0. This is because handling
      // constants other than 0 in indexing map, may cause problems in the
      // lowering pipeline later.
      if (constantIndex.getLimitedValue() != 0)
        return failure();
      exprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
      continue;
    }
    // 2. The value is defined outside the body.
    if (!linalgOp->isAncestor(indexValue.getParentRegion()->getParentOp())) {
      // Index the tensor along this dimension by using tensor.extract_slice.
      SmallVector<OpFoldResult> base(sourceShape.size(),
                                     rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(
          rewriter, extractOp.getLoc(), extractOp.getTensor());
      SmallVector<OpFoldResult> strides(sourceShape.size(),
                                        rewriter.getIndexAttr(1));
      // Index on the loop invariant value.
      base[idx] = indexValue;
      sizes[idx] = rewriter.getIndexAttr(1);
      rewriter.create<tensor::ExtractSliceOp>(extractOp.getLoc());
    }
    // 3. The indexing value is a linalg.index.
    if (auto indexOp = indexValue.getDefiningOp<linalg::IndexOp>()) {
      // Make sure that for this index, the size of the input and output
      // match and are not dynamic. We need this to maintain the op to be
      // elementwise.
      // TODO: This restriction can be relaxed by adding a extract_slice op
      // on the `source` tensor. This is not same as raising the whole
      // operation to an extract_slice, as there can be permutations and
      // projections involved.
      if (ShapedType::isDynamic(sourceShape[idx]) ||
          ShapedType::isDynamic(resultShape[indexOp.getDim()]) ||
          sourceShape[idx] != resultShape[indexOp.getDim()]) {
        return failure();
      }
      exprs.push_back(
          getAffineDimExpr(indexOp.getDim(), rewriter.getContext()));
      continue;
    }
    return failure();
  }
  AffineMap indexingMap = AffineMap::get(
      /*dimCount=*/linalgOp.getNumLoops(),
      /*symbolCount=*/0, exprs, rewriter.getContext());

  // Replace the linalgOp with a new linalgOp where the source tensor is
  // an input with the indexing map.
  SmallVector<Value> newInputs = linalgOp.getInputs();
  newInputs.insert(newInputs.begin(), extractOp.getTensor());
  SmallVector<Attribute> newIndexingMaps;
  newIndexingMaps.push_back(AffineMapAttr::get(indexingMap));
  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    newIndexingMaps.push_back(AffineMapAttr::get(map));
  }

  auto bodyBuilder = [&](OpBuilder &builder, Location loc, ValueRange args) {
    // Create an IR mapping from old block arguements to new ones.
    IRMapping mapper;
    ArrayRef<BlockArgument> oldArgs = linalgOp.getBody()->getArguments();
    // Map i^th old argument to (i + 1)^th new argument.
    for (unsigned i = 0; i < oldArgs.size(); ++i) {
      mapper.map(oldArgs[i], args[i + 1]);
    }
    // Clone the body of the linalgOp.
    for (Operation &op : linalgOp.getBody()->getOperations()) {
      // Replace the extractOp with the first block argument.
      if (&op == extractOp) {
        mapper.map(op.getResult(0), args[0]);
      } else {
        builder.clone(op, mapper);
      }
    }
  };

  linalg::GenericOp newLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
      linalgOp.getOutputs(),
      ArrayAttr::get(linalgOp->getContext(), newIndexingMaps),
      linalgOp.getIteratorTypesAttr(), linalgOp.getDocAttr(),
      linalgOp.getLibraryCallAttr(), bodyBuilder);

  return newLinalgOp;
}

struct RaiseTensorExtractToInput final : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> newOp =
        raiseTensorExtractToInput(genericOp, rewriter);
    if (failed(newOp)) {
      return failure();
    }
    rewriter.replaceOp(genericOp, newOp.value());
    return success();
  }
};

} // namespace

void populateTensorExtractToInputRaisingPatterns(RewritePatternSet &patterns) {
  patterns.add<RaiseTensorExtractToInput>(patterns.getContext());
}

namespace {
struct RaiseSpecialOpsPass final
    : impl::RaiseSpecialOpsPassBase<RaiseSpecialOpsPass> {
  using impl::RaiseSpecialOpsPassBase<
      RaiseSpecialOpsPass>::RaiseSpecialOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTensorExtractToInputRaisingPatterns(patterns);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
