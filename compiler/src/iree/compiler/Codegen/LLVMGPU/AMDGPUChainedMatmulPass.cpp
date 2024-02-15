// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

namespace {

struct AMDGPUPrepareForChainedMatmulPass
    : public AMDGPUPrepareForChainedMatmulBase<
          AMDGPUPrepareForChainedMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  /// Given a vector contract of the form
  /// %output = vector.contract %lhs, %rhs, %acc
  /// this function swaps the operands (%rhs, %lhs),
  /// transposes the accumulator and output and updates
  /// the indexing maps for the new contract op.
  void swapOperandsAndTranspose(RewriterBase &rewriter,
                                vector::ContractionOp contractOp) const {
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    Value acc = contractOp.getAcc();
    rewriter.setInsertionPoint(contractOp);
    Value transposed = rewriter.create<vector::TransposeOp>(
        contractOp.getLoc(), acc, SmallVector<int64_t>{1, 0});
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{n, k}, {m, k}, {n, m}});
    vector::ContractionOp swappedOp = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), rhs, lhs, transposed,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        contractOp.getIteratorTypesAttr());
    Value newResult = swappedOp.getResult();
    transposed = rewriter.create<vector::TransposeOp>(
        contractOp.getLoc(), newResult, SmallVector<int64_t>{1, 0});
    rewriter.replaceOp(contractOp, transposed);
  }

  /// The only compatible indexing map corresponds to
  /// the matmul_transpose_b, and is
  /// (m, n, k) -> (m, k)
  /// (m, n, k) -> (n, k)
  /// (m, n, k) -> (m, n)
  ///
  /// We only check for matmul_transpose_b because transposing the inputs of
  /// the matmul for it still produces a matmul_transpose_b, which is good
  /// for reads. We could probably do this for other matmuls also, but
  /// we need to think if we want to enable that.
  bool isCompatibleIndexingMap(vector::ContractionOp contractOp,
                               MLIRContext *ctx) const {
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{m, k}, {n, k}, {m, n}});
    return newIndexingMaps == contractOp.getIndexingMapsArray();
  }

  /// Returns the first matmul in a matmul chain.
  ///
  /// A chained matmul is one where the lhs of the candidate matrix
  /// is a result of another matmul (a matmul lies in the backward slice of lhs
  /// of the first matmul).
  FailureOr<vector::ContractionOp>
  getChainParent(vector::ContractionOp candidate) const {
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(candidate.getLhs(), &backwardSlice);
    for (Operation *sliceOp : backwardSlice) {
      auto chainParent = dyn_cast<vector::ContractionOp>(sliceOp);
      if (!chainParent) {
        continue;
      }

      // Check if the chainParent is a compatible matmul.
      if (isCompatibleIndexingMap(chainParent, candidate.getContext())) {
        return chainParent;
      }
    }
    return failure();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<vector::ContractionOp> matmulCandidates;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      matmulCandidates.push_back(contractOp);
    });

    IRRewriter rewriter(funcOp.getContext());
    for (auto candidate : matmulCandidates) {
      FailureOr<vector::ContractionOp> maybeChainedParent =
          getChainParent(candidate);
      if (failed(maybeChainedParent)) {
        continue;
      }
      auto chainParent = maybeChainedParent.value();
      swapOperandsAndTranspose(rewriter, chainParent);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createAMDGPUPrepareForChainedMatmulPass() {
  return std::make_unique<AMDGPUPrepareForChainedMatmulPass>();
}

} // namespace mlir::iree_compiler
