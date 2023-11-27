// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep
                                                                     //
struct IREEVectorExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEVectorExtDialect::initialize() {
  addInterfaces<IREEVectorExtDialectOpAsmInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc"
      >();
}

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.cpp.inc"

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrInterfaces.cpp.inc"

bool PerDimLayoutAttr::contains(const LayoutDimension &dim) {
  for (LayoutDimensionAttr label : getLabels()) {
    if (label.getValue() == dim)
      return true;
  }
  return false;
}

std::optional<int64_t> PerDimLayoutAttr::getShape(const LayoutDimension &dim) {
  for (auto value : llvm::zip(getLabels(), getShapes())) {
    if (dim == std::get<0>(value).getValue())
      return std::get<1>(value);
  }
  return std::nullopt;
}

// Get the SIMT Vector shape in the order specified by dims. If no dims are
// specified, then return an empty vector.
SmallVector<int64_t>
LayoutAttr::getSIMTVectorShape(ArrayRef<LayoutDimension> dims) {
  SmallVector<int64_t> simtVectorShape;
  for (LayoutDimension dim : dims) {
    ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
    for (PerDimLayoutAttr layout : layouts) {
      if (!layout.contains(dim))
        continue;
      simtVectorShape.push_back(layout.getShape(dim).value());
    }
  }
  return simtVectorShape;
}

bool LayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  for (auto perDimLayout : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = perDimLayout.value().getShapes();
    int64_t computedShape = std::reduce(layoutShape.begin(), layoutShape.end(),
                                        1, std::multiplies<int64_t>());
    int64_t expectedShape = shape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return false;
    }
  }
  return true;
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> projectedDims) const {
  assert(projectedDims.size() == getLayouts().size() &&
         "projectedDims size must match layout size");
  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(projectedDims.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto pair : llvm::zip(projectedDims, layouts)) {
    if (!std::get<0>(pair))
      newLayouts.push_back(std::get<1>(pair));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// Permute the layout according to the provided permutation
// vector. The dimensionality of the layout remains the same.
VectorLayoutInterface LayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == getLayouts().size() &&
         "permutation size must match layout size");
  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(permutation.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (unsigned index : permutation) {
    assert(index >= 0 && index < layouts.size());
    newLayouts.push_back(layouts[index]);
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

bool BlockLayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  return true;
}

BlockLayoutAttr BlockLayoutAttr::project(ArrayRef<bool> projectedDims) const {
  // Project the given dim in each field.
  SmallVector<int64_t> newBatch;
  SmallVector<int64_t> newDistributed;
  SmallVector<int64_t> newThread;
  for (auto [index, projected] : llvm::enumerate(projectedDims)) {
    if (!projected) {
      newBatch.push_back(getBatch()[index]);
      newDistributed.push_back(getDistributed()[index]);
      newThread.push_back(getThread()[index]);
    }
  }
  return BlockLayoutAttr::get(getContext(), newBatch, newDistributed,
                              newThread);
}

BlockLayoutAttr BlockLayoutAttr::permute(ArrayRef<unsigned> permutation) const {
  // Permute the given dim in each field.
  SmallVector<int64_t> newBatch;
  SmallVector<int64_t> newDistributed;
  SmallVector<int64_t> newThread;
  for (unsigned index : permutation) {
    newBatch.push_back(getBatch()[index]);
    newDistributed.push_back(getDistributed()[index]);
    newThread.push_back(getThread()[index]);
  }
  return BlockLayoutAttr::get(getContext(), newBatch, newDistributed,
                              newThread);
}

static void forAll(ArrayRef<int64_t> iterates,
                   std::function<void(ArrayRef<int64_t>)> callback) {

  SmallVector<int64_t> curr(iterates.size(), 0);

  // Iterate over all possible batch tiles.
  while (true) {
    callback(curr);

    // Increment the current tile.
    bool done = false;
    for (int64_t i = curr.size() - 1; i >= 0; --i) {
      if (curr[i] < iterates[i] - 1) {
        ++curr[i];
        done = true;
        break;
      }
      curr[i] = 0;
    }
    if (!done)
      break;
  }
}

void BlockLayoutAttr::forAllBatchTiles(
    ArrayRef<int64_t> vectorShape,
    std::function<void(ArrayRef<int64_t>)> callback) const {
  SmallVector<int64_t> batchIterates;
  for (int64_t i = 0; i < vectorShape.size(); ++i) {
    batchIterates.push_back(
        vectorShape[i] /
        (getBatch()[i] * getDistributed()[i] * getThread()[i]));
  }
  forAll(batchIterates, callback);
}

void BlockLayoutAttr::forAllDistributedTiles(
    std::function<void(ArrayRef<int64_t>)> callback) const {
  forAll(getBatch(), callback);
}

void BlockLayoutAttr::forAllThreadTiles(
    std::function<void(ArrayRef<int64_t>)> callback) const {
  forAll(getDistributed(), callback);
}

void BlockLayoutAttr::forAllElementsInThreadTile(
    std::function<void(ArrayRef<int64_t>)> callback) const {
  forAll(getThread(), callback);
}

SmallVector<int64_t> BlockLayoutAttr::getBatchSize() const {
  SmallVector<int64_t> batchSize;
  for (int64_t i = 0; i < getBatch().size(); ++i) {
    batchSize.push_back(getBatch()[i] * getDistributed()[i] * getThread()[i]);
  }
  return batchSize;
}

SmallVector<int64_t> BlockLayoutAttr::getDistributedSize() const {
  SmallVector<int64_t> distributedSize;
  for (int64_t i = 0; i < getDistributed().size(); ++i) {
    distributedSize.push_back(getDistributed()[i] * getThread()[i]);
  }
  return distributedSize;
}

SmallVector<int64_t> BlockLayoutAttr::getThreadSize() const {
  SmallVector<int64_t> threadSize;
  for (int64_t i = 0; i < getThread().size(); ++i) {
    threadSize.push_back(getThread()[i]);
  }
  return threadSize;
}

SmallVector<int64_t>
BlockLayoutAttr::getNumBatches(ArrayRef<int64_t> vectorShape) const {
  SmallVector<int64_t> numBatches;
  SmallVector<int64_t> batchSize = getBatchSize();
  for (int64_t i = 0; i < vectorShape.size(); ++i) {
    numBatches.push_back(vectorShape[i] / batchSize[i]);
  }
  return numBatches;
}
