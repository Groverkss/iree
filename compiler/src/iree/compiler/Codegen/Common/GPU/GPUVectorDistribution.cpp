// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

using VectorValue = TypedValue<VectorType>;

static const char *kVectorLayoutFetcherStorageAttrName =
    "__vector_layout_fetcher_storage";

static void setOpSignature(Operation *op, VectorLayoutAnalysis &analysis) {
  SmallVector<Attribute> operands;
  SmallVector<Attribute> results;

  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
      operands.push_back(
          analysis.getLayout<VectorLayoutInterface>(vectorOperand));
      continue;
    }
    operands.push_back(VectorLayoutInterface());
  }

  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<VectorValue>(result)) {
      results.push_back(
          analysis.getLayout<VectorLayoutInterface>(vectorResult));
      continue;
    }
    results.push_back(VectorLayoutInterface());
  }

  ArrayAttr operandsAttr = ArrayAttr::get(op->getContext(), operands);
  ArrayAttr resultsAttr = ArrayAttr::get(op->getContext(), results);
  SmallVector<Attribute> signature = {operandsAttr, resultsAttr};
  op->setAttr(kVectorLayoutFetcherStorageAttrName,
              ArrayAttr::get(op->getContext(), signature));
}

static bool hasOpSignature(Operation *op) {
  return op->hasAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
}

static DistributionSignature getOpSignature(Operation *op) {
  ArrayAttr signatureAttr =
      op->getAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
  assert(signatureAttr && "Op should have a signature attribute.");
  assert(signatureAttr.size() == 2 && "Malformed signature attribute.");

  ArrayAttr operandsAttr = dyn_cast<ArrayAttr>(signatureAttr[0]);
  ArrayAttr resultsAttr = dyn_cast<ArrayAttr>(signatureAttr[1]);
  assert(operandsAttr && resultsAttr && "Malformed signature attribute.");

  DistributionSignature signature;
  for (Attribute operandAttr : operandsAttr) {
    auto operandLayout = cast<VectorLayoutInterface>(operandAttr);
    assert(operandLayout && "Malformed signature attribute.");
    signature.operands.push_back(operandLayout);
  }

  for (Attribute resultAttr : resultsAttr) {
    VectorLayoutInterface resultLayout =
        cast<VectorLayoutInterface>(resultAttr);
    assert(resultLayout && "Malformed signature attribute.");
    signature.results.push_back(resultLayout);
  }

  return signature;
}

VectorValue
DistributionPattern::getDistributed(RewriterBase &rewriter, VectorValue value,
                                    VectorLayoutInterface layout) const {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    return cast<VectorValue>(toSIMD.getInput());
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape =
      options.getDistributedShape(value, layout);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void DistributionPattern::replaceOpWithDistributedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<VectorValue>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Add to replacements.
      replacement = toSIMD;
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

std::optional<DistributionSignature>
DistributionPattern::getOpSignature(Operation *op) const {
  if (!hasOpSignature(op)) {
    return std::nullopt;
  }
  return ::mlir::iree_compiler::getOpSignature(op);
}

LogicalResult DistributionPattern::match(Operation *op) const {
  if (!hasOpSignature(op)) {
    return failure();
  }
  return match(op);
}

class DistributeConstants : public OpDistributionPattern<arith::ConstantOp> {
public:
  using OpDistributionPattern<arith::ConstantOp>::OpDistributionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Value constantResult = constantOp.getResult();
    if (!isa<VectorType>(constantResult.getType()))
      return failure();
    auto constant = cast<VectorValue>(constantResult);

    // Only handle splat values for now.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature.results[0];

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType = VectorType::get(
        options.getDistributedShape(constant, layout), elementType);
    Operation *distirbutedOp = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), vectorType, attr.getSplatValue<Attribute>());
    replaceOpWithDistributedValues(rewriter, constantOp,
                                   distirbutedOp->getResult(0));
    return success();
  }
};

template <typename OpTy>
class DistributeElementwise : public OpDistributionPattern<OpTy> {
public:
  using OpDistributionPattern<OpTy>::OpDistributionPattern;

  LogicalResult matchAndRewrite(OpTy op, DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Get the distributed operands.
    SmallVector<Value> operands;
    for (auto [operand, opLayout] :
         llvm::zip(op->getOperands(), signature.operands)) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      opLayout);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (auto [result, resLayout] :
         llvm::zip(op->getResults(), signature.results)) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        resultType = VectorType::get(
            options.getDistributedShape(vectorResult, resLayout),
            vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), operands, resultTypes);
    DistributionPattern::replaceOpWithDistributedValues(
        rewriter, op, distributedOp->getResults());
    return success();
  }
};

static void
debugPrintUniqueOperationNames(SmallVectorImpl<Operation *> &worklist) {
  DenseSet<StringRef> uniqueNames;
  for (Operation *op : worklist) {
    uniqueNames.insert(op->getName().getStringRef());
  }

  for (StringRef name : uniqueNames) {
    llvm::dbgs().indent(2) << "* " << name << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

/// A rewriter for the pattern rewriting driver.
class VectorDistributionRewriter : public PatternRewriter {
public:
  VectorDistributionRewriter(MLIRContext *ctx) : PatternRewriter(context) {}
};

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns,
                                    VectorLayoutOptions &options) {

  SmallVector<Operation *> worklist;

  VectorDistributionRewriter rewriter(root->getContext());
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  LLVM_DEBUG(llvm::dbgs() << "Collecting operations to be distributed\n");
  root->walk([&](Operation *op) {
    if (hasOpSignature(op)) {
      worklist.push_back(op);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Operations to be distributed:\n");
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist));

  // Note that the pattern application here never runs on a newly created
  // operation. It always runs on an existing operation. This ensures that no
  // invalidated state of the analysis is ever used.
  for (Operation *op : worklist) {
    LLVM_DEBUG(llvm::dbgs() << "Distributing: ");
    LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << ": Failed to distribute operation:\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << ": Successfully distributed operation:\n");
  }
}

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  auto values = llvm::to_vector_of<Value>(op->getOperands());
  llvm::append_range(values, op->getResults());

  // Check if all operands and results of this operation have a layout.
  return llvm::all_of(values, [&](Value value) -> bool {
    if (auto vectorValue = dyn_cast<VectorValue>(value)) {
      return (bool)analysis.getLayout<Attribute>(vectorValue);
    }
    return false;
  });
}

void distributeVectorOps(Operation *root, VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");
  VectorLayoutAnalysis analysis(root);
  options.setAnchorOps(analysis);
  if (failed(analysis.run()))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Succeded\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Go to each operation, and set its distribution signature.
  LLVM_DEBUG(
      llvm::dbgs() << "Setting distribution signatures for operations\n");
  root->walk([&](Operation *op) {
    if (canDistribute(op, analysis)) {
      setOpSignature(op, analysis);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Distribution signatures set\n");
  LLVM_DEBUG(root->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Run the distribution patterns.
  RewritePatternSet patterns(root->getContext());
  patterns.add<DistributeConstants>(root->getContext(), options);

  patterns.add<DistributeElementwise<arith::AddFOp>,
               DistributeElementwise<arith::AddIOp>,
               DistributeElementwise<arith::MulFOp>,
               DistributeElementwise<arith::MulIOp>>(root->getContext(),
                                                     options);

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  return applyVectorDistribution(root, frozenPatterns, options);
}

} // namespace mlir::iree_compiler
