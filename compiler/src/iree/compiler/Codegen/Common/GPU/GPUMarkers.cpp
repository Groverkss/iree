// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUMarkers.h"

namespace mlir::iree_compiler::gpu_markers {

const char *kBasisSuffix = "basis";

static std::string getBasisLevelName(IREE::GPU::TilingLevel level) {
  return std::string(IREE::GPU::getTilingLevelName(level)) + "_" + kBasisSuffix;
}

void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, ArrayRef<int64_t> basis,
              ArrayRef<int64_t> mapping) {
  Builder b(context);
  ArrayAttr basisAttr = b.getArrayAttr(
      {b.getDenseI64ArrayAttr(basis), b.getDenseI64ArrayAttr(mapping)});
  attrs.emplace_back(b.getNamedAttr(getBasisLevelName(level), basisAttr));
}

LogicalResult getBasis(IREE::GPU::LoweringConfigAttr config,
                       IREE::GPU::TilingLevel level,
                       SmallVector<int64_t> &basis,
                       SmallVector<int64_t> &mapping) {
  auto basisAttr = dyn_cast_or_null<ArrayAttr>(
      config.getAttributes().get(getBasisLevelName(level)));
  if (!basisAttr) {
    return failure();
  }

  ArrayRef<Attribute> attrs = basisAttr.getValue();
  if (attrs.size() != 2) {
    return failure();
  }

  DenseI64ArrayAttr basisArray = dyn_cast_or_null<DenseI64ArrayAttr>(attrs[0]);
  DenseI64ArrayAttr mappingArray =
      dyn_cast_or_null<DenseI64ArrayAttr>(attrs[1]);

  if (!basisArray || !mappingArray) {
    return failure();
  }

  basis = SmallVector<int64_t>(basisArray.asArrayRef());
  mapping = SmallVector<int64_t>(mappingArray.asArrayRef());

  return success();
}

} // namespace mlir::iree_compiler::gpu_markers
