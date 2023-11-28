// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"

using namespace mlir::iree_compiler::IREE::VectorExt;
using namespace mlir::iree_compiler;
using namespace mlir;

enum class MMAType {
  NONE = 0,
  M16N8K16 = 1,
};

enum class MMAMatrixType { AMatrix, BMatrix, CMatrix };

static MMAType getMMAType(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          ArrayRef<int64_t> cShape) {

  if (aShape.size() != 2 || bShape.size() != 2 || cShape.size() != 2)
    return MMAType::NONE;

  // M --> 16
  // N --> 8
  // K --> 16
  // Assuming A is MxK, B is NxK, C is MxN
  if (aShape[0] % 16 == 0 && aShape[1] % 16 == 0) {
    if (bShape[0] % 8 == 0 && bShape[1] % 16 == 0) {
      if (cShape[0] % 16 == 0 && cShape[1] % 8 == 0) {
        return MMAType::M16N8K16;
      }
    }
  }

  return MMAType::NONE;
}

static SmallVector<int64_t> multiplyTiles(ArrayRef<int64_t> tilesA,
                                          ArrayRef<int64_t> tilesB) {
  // TODO: Generalize.
  return {tilesA[0] * tilesB[0], tilesA[1] * tilesB[1]};
}

static SmallVector<int64_t> divideTiles(ArrayRef<int64_t> tilesA,
                                        ArrayRef<int64_t> tilesB) {
  // TODO: Generalize.
  return {tilesA[0] / tilesB[0], tilesA[1] / tilesB[1]};
}

static std::array<int64_t, 2> getMMACanonicalShape(MMAType mmaType,
                                                   MMAMatrixType matrixType) {
  switch (mmaType) {
  case MMAType::M16N8K16:
    switch (matrixType) {
    case MMAMatrixType::AMatrix:
      return {16, 16};
    case MMAMatrixType::BMatrix:
      return {8, 16};
    case MMAMatrixType::CMatrix:
      return {16, 8};
    }
    return {};
  default:
    return {};
  }
}

static bool isMatmulTransposeB(vector::ContractionOp contractOp) {
  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(contractOp.getContext(), m, n, k);
  auto iteratorTypes = contractOp.getIteratorTypes().getValue();
  if (!(vector::isParallelIterator(iteratorTypes[0]) &&
        vector::isParallelIterator(iteratorTypes[1]) &&
        vector::isReductionIterator(iteratorTypes[2])))
    return false;
  SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
  return maps == infer({{m, k}, {n, k}, {m, n}});
}

static void setAnchorOps(VectorLayoutAnalysis &analysis, Operation *op) {
  op->walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (!isMatmulTransposeB(contractOp))
        return WalkResult::advance();
      MLIRContext *ctx = contractOp.getContext();

      if (!isa<VectorType>(contractOp.getAccType())) {
        return WalkResult::advance();
      }

      ArrayRef<int64_t> aShape = contractOp.getLhsType().getShape();
      ArrayRef<int64_t> bShape = contractOp.getRhsType().getShape();
      ArrayRef<int64_t> cShape =
          cast<VectorType>(contractOp.getAccType()).getShape();

      MMAType mmaType = getMMAType(aShape, bShape, cShape);
      if (mmaType == MMAType::NONE)
        return WalkResult::advance();

      std::array<int64_t, 2> threadTile = {1, 2};

      std::array<int64_t, 2> distributedTile = {8, 4};

      std::array<int64_t, 2> canonicalShapeA =
          getMMACanonicalShape(mmaType, MMAMatrixType::AMatrix);
      std::array<int64_t, 2> canonicalShapeB =
          getMMACanonicalShape(mmaType, MMAMatrixType::BMatrix);
      std::array<int64_t, 2> canonicalShapeC =
          getMMACanonicalShape(mmaType, MMAMatrixType::CMatrix);

      SmallVector<int64_t> batchTileA = divideTiles(
          canonicalShapeA, multiplyTiles(distributedTile, threadTile));
      SmallVector<int64_t> batchTileB = divideTiles(
          canonicalShapeB, multiplyTiles(distributedTile, threadTile));
      SmallVector<int64_t> batchTileC = divideTiles(
          canonicalShapeC, multiplyTiles(distributedTile, threadTile));

      // --------- A-matrix
      BlockLayoutAttr nvidiaMMASyncLayoutA =
          BlockLayoutAttr::get(ctx, batchTileA, distributedTile, threadTile);

      // --------- B-matrix
      BlockLayoutAttr nvidiaMMASyncLayoutB =
          BlockLayoutAttr::get(ctx, batchTileB, distributedTile, threadTile);

      // --------- C-matrix
      BlockLayoutAttr nvidiaMMASyncLayoutC =
          BlockLayoutAttr::get(ctx, batchTileC, distributedTile, threadTile);

      // Set layout for A and B matrix.
      analysis.setAnchor(contractOp.getLhs(), nvidiaMMASyncLayoutA);
      analysis.setAnchor(contractOp.getRhs(), nvidiaMMASyncLayoutB);
      // Result and accumulator have the same layout: C-matrix
      analysis.setAnchor(contractOp.getResult(), nvidiaMMASyncLayoutC);
      analysis.setAnchor(contractOp.getAcc(), nvidiaMMASyncLayoutC);
    }

    return WalkResult::advance();
  });
}

void NVIDIALayoutProvider::setAnchorOps() { ::setAnchorOps(analysis, root); }
