// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

class LayoutProvider {
public:
  LayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : analysis(analysis), root(root) {}

  virtual ~LayoutProvider() = default;

  virtual void setAnchorOps() = 0;

protected:
  VectorLayoutAnalysis &analysis;
  Operation *root;
};

class NVIDIALayoutProvider : public LayoutProvider {
public:
  NVIDIALayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : LayoutProvider(analysis, root) {}

  void setAnchorOps() override;
};

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_
