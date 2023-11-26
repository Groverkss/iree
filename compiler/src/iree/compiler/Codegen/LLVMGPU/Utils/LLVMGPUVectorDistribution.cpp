// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
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

#define DEBUG_TYPE "iree-llvmgpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

enum class MMAType {
  M16N8K16,
  NONE,
};

enum class MMAMatrixType { AMatrix, BMatrix, CMatrix };

static MMAType getMMAType(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          ArrayRef<int64_t> cShape) {
  if ((aShape[0] % 16 == 0) && (aShape[1] % 16 == 0) && (cShape[0] % 16 == 0) &&
      (cShape[1] % 8 == 0)) {
    if ((bShape[0] % 16 == 0) && (bShape[1] % 8 == 0))
      return MMAType::M16N8K16;
  }
  return MMAType::NONE;
}

static std::array<int, 2> getMMACanonicalShape(MMAType mmaType,
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
          cast<VectorType>(contractOp.getResultType()).getShape();

      MMAType mmaType = getMMAType(aShape, bShape, cShape);
      if (mmaType == MMAType::NONE)
        return WalkResult::advance();

      auto constructLayout = [&](int64_t canonicalShape, int64_t vectorShape,
                                 ArrayRef<LayoutDimension> dims,
                                 SmallVectorImpl<int64_t> &shapes) {
        int64_t batchDim = vectorShape / canonicalShape;
        shapes.insert(shapes.begin(), batchDim);
        SmallVector<LayoutDimensionAttr> labels;
        auto toAttr = [&](LayoutDimension dim) {
          return LayoutDimensionAttr::get(ctx, dim);
        };
        std::transform(dims.begin(), dims.end(), std::back_inserter(labels),
                       toAttr);
        return PerDimLayoutAttr::get(ctx, labels, shapes);
      };

      SmallVector<LayoutDimension> dims;
      SmallVector<int64_t> shapes;

      std::array<int, 2> canonicalShapeA =
          getMMACanonicalShape(mmaType, MMAMatrixType::AMatrix);
      std::array<int, 2> canonicalShapeB =
          getMMACanonicalShape(mmaType, MMAMatrixType::BMatrix);
      std::array<int, 2> canonicalShapeC =
          getMMACanonicalShape(mmaType, MMAMatrixType::CMatrix);

      // --------- A-matrix
      dims = {LayoutDimension::BATCHX, LayoutDimension::LANEY,
              LayoutDimension::VECTORZ};
      shapes = {8, 2};
      PerDimLayoutAttr rowLayout =
          constructLayout(canonicalShapeA[0], aShape[0], dims, shapes);

      dims = {LayoutDimension::BATCHY, LayoutDimension::VECTORX,
              LayoutDimension::LANEX, LayoutDimension::VECTORY};
      shapes = {2, 4, 2};
      PerDimLayoutAttr colLayout =
          constructLayout(canonicalShapeA[1], aShape[1], dims, shapes);

      SmallVector<PerDimLayoutAttr> layouts;
      layouts.push_back(rowLayout);
      layouts.push_back(colLayout);

      LayoutAttr nvidiaMMASyncLayoutA = LayoutAttr::get(ctx, layouts);

      // --------- B-matrix
      dims = {LayoutDimension::BATCHX, LayoutDimension::LANEY,
              LayoutDimension::VECTORZ};
      shapes = {8, 1};
      rowLayout = constructLayout(canonicalShapeB[0], bShape[0], dims, shapes);

      dims = {LayoutDimension::BATCHY, LayoutDimension::VECTORX,
              LayoutDimension::LANEX, LayoutDimension::VECTORY};
      shapes = {2, 4, 2};
      colLayout = constructLayout(canonicalShapeB[1], bShape[1], dims, shapes);

      layouts.clear();
      layouts.push_back(rowLayout);
      layouts.push_back(colLayout);

      LayoutAttr nvidiaMMASyncLayoutB = LayoutAttr::get(ctx, layouts);

      // --------- C-matrix
      dims = {LayoutDimension::BATCHX, LayoutDimension::LANEY,
              LayoutDimension::VECTORY};
      shapes = {8, 2};
      rowLayout = constructLayout(canonicalShapeC[0], cShape[0], dims, shapes);

      dims = {LayoutDimension::BATCHY, LayoutDimension::VECTORX,
              LayoutDimension::LANEX, LayoutDimension::VECTORZ};
      shapes = {2, 4, 1};
      colLayout = constructLayout(canonicalShapeC[1], cShape[1], dims, shapes);

      layouts.clear();
      layouts.push_back(rowLayout);
      layouts.push_back(colLayout);

      LayoutAttr nvidiaMMASyncLayoutC = LayoutAttr::get(ctx, layouts);

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

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  setAnchorOps(analysis, funcOp);
  if (failed(analysis.run())) {
    funcOp.emitError("layout analysis failed");
    return;
  }

  funcOp.walk([&](Operation *op) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      HighDimLayout layout = analysis.getLayout(operand);
      if (layout) {
        op->setAttr(("layout" + std::to_string(index)), layout);
      }
    }
  });

  return;
}

} // namespace mlir::iree_compiler
