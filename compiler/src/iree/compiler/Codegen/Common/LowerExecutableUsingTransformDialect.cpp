// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"

namespace mlir::iree_compiler {

namespace {
class LowerExecutableUsingTransformDialectPass
    : public LowerExecutableUsingTransformDialectBase<
          LowerExecutableUsingTransformDialectPass> {
public:
  void runOnOperation() override;
};
} // namespace

void LowerExecutableUsingTransformDialectPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto funcOps = moduleOp.getOps<FunctionOpInterface>();

  if (funcOps.empty() || !llvm::hasSingleElement(funcOps)) {
    // Can only handle dispatches with single functions on the transform dialect
    // path.
    return;
  }

  auto funcOp = *funcOps.begin();
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo || translationInfo.getDispatchLoweringPassPipeline() !=
                              IREE::Codegen::DispatchLoweringPassPipeline::
                                  TransformDialectCodegen) {
    return;
  }

  SymbolRefAttr codegenSpec = translationInfo.getCodegenSpec();
  StringRef entryPoint =
      codegenSpec ? codegenSpec.getLeafReference() : StringRef("");
  OpPassManager modulePassManager(ModuleOp::getOperationName());
  modulePassManager.addPass(
      iree_compiler::createTransformDialectInterpreterPass(entryPoint));
  modulePassManager.addPass(createDropSchedulePass());
  if (failed(runPipeline(modulePassManager, moduleOp))) {
    moduleOp.emitOpError("failed to run transform dialect passes");
    return signalPassFailure();
  }

  // Set translation info to be none to avoid subsequent codegen pass pipelines.
  auto translationInfoModified = getTranslationInfo(funcOp);
  if (!translationInfoModified ||
      translationInfoModified.getDispatchLoweringPassPipeline() !=
          IREE::Codegen::DispatchLoweringPassPipeline::None) {
    funcOp->emitOpError("expected transform dialect lowering to set the "
                        "translation_info to use None");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLowerExecutableUsingTransformDialectPass() {
  return std::make_unique<LowerExecutableUsingTransformDialectPass>();
}

} // namespace mlir::iree_compiler
