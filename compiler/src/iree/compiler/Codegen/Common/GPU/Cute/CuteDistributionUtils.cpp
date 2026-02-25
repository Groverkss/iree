// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Cute/CuteDistributionUtils.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

using IREE::Map::filterLeafInfos;
using IREE::Map::LeafInfo;
using IREE::Map::makeLeaf;
using IREE::Map::makeTuple;
using IREE::Map::PackLayoutAttr;

PackLayoutAttr
buildPackLayout(MLIRContext *ctx, ArrayRef<int64_t> subgroupTile,
                ArrayRef<int64_t> batchTile, ArrayRef<int64_t> outerTile,
                ArrayRef<int64_t> threadTile, ArrayRef<int64_t> elementTile,
                ArrayRef<int64_t> subgroupStrides,
                ArrayRef<int64_t> threadStrides, int64_t subgroupSize) {

  int64_t rank = subgroupTile.size();
  SmallVector<Attribute> modeShapes, modeStrides;

  for (int64_t i = 0; i < rank; ++i) {
    // Build leaves outermost → innermost (left → right in lex convention).
    // Skip size-1 leaves since they don't affect data strides or distribution.
    SmallVector<Attribute> shapes, strides;

    if (subgroupTile[i] > 1) {
      shapes.push_back(makeLeaf(ctx, subgroupTile[i]));
      int64_t stride =
          subgroupStrides[i] == 0 ? 0 : subgroupStrides[i] * subgroupSize;
      strides.push_back(makeLeaf(ctx, stride));
    }

    if (batchTile[i] > 1) {
      shapes.push_back(makeLeaf(ctx, batchTile[i]));
      strides.push_back(makeLeaf(ctx, 0));
    }

    if (outerTile[i] > 1) {
      shapes.push_back(makeLeaf(ctx, outerTile[i]));
      strides.push_back(makeLeaf(ctx, 0));
    }

    if (threadTile[i] > 1) {
      shapes.push_back(makeLeaf(ctx, threadTile[i]));
      strides.push_back(makeLeaf(ctx, threadStrides[i]));
    }

    if (elementTile[i] > 1) {
      shapes.push_back(makeLeaf(ctx, elementTile[i]));
      strides.push_back(makeLeaf(ctx, 0));
    }

    // Degenerate case: all tiles are 1.
    if (shapes.empty()) {
      shapes.push_back(makeLeaf(ctx, 1));
      strides.push_back(makeLeaf(ctx, 0));
    }

    if (shapes.size() == 1) {
      modeShapes.push_back(shapes[0]);
      modeStrides.push_back(strides[0]);
    } else {
      modeShapes.push_back(makeTuple(ctx, shapes));
      modeStrides.push_back(makeTuple(ctx, strides));
    }
  }

  return PackLayoutAttr::get(ctx, makeTuple(ctx, modeShapes),
                             makeTuple(ctx, modeStrides));
}

SmallVector<LeafDimInfo> getLeafDimMap(PackLayoutAttr layout) {
  SmallVector<LeafDimInfo> map;
  int32_t rank = layout.getRank();
  for (int32_t d = 0; d < rank; ++d) {
    auto valLeaves =
        filterLeafInfos(layout.getShapeMode(d), layout.getStrideMode(d),
                        [](const LeafInfo &l) { return l.stride == 0; });
    if (valLeaves.empty()) {
      map.push_back({d, 1});
    } else {
      for (auto &leaf : valLeaves) {
        map.push_back({d, static_cast<int64_t>(leaf.dataStride)});
      }
    }
  }
  return map;
}

SmallVector<Value> buildThreadOffsets(OpBuilder &b, Location loc,
                                      PackLayoutAttr layout, Value threadId) {
  int32_t rank = layout.getRank();
  SmallVector<Value> offsets;

  for (int32_t d = 0; d < rank; ++d) {
    auto threadLeaves =
        filterLeafInfos(layout.getShapeMode(d), layout.getStrideMode(d),
                        [](const LeafInfo &l) { return l.stride > 0; });

    if (threadLeaves.empty()) {
      offsets.push_back(arith::ConstantIndexOp::create(b, loc, 0));
      continue;
    }

    Value sum = nullptr;
    for (auto &leaf : threadLeaves) {
      // Extract this leaf's thread coordinate: (tid / tidStride) % size
      Value coord = threadId;
      if (leaf.stride != 1) {
        Value s = arith::ConstantIndexOp::create(b, loc, leaf.stride);
        coord = arith::DivUIOp::create(b, loc, coord, s);
      }
      Value sz = arith::ConstantIndexOp::create(b, loc, leaf.size);
      coord = arith::RemUIOp::create(b, loc, coord, sz);

      // Multiply by data stride to get contribution.
      if (leaf.dataStride != 1) {
        Value ds = arith::ConstantIndexOp::create(b, loc, leaf.dataStride);
        coord = arith::MulIOp::create(b, loc, coord, ds);
      }

      if (!sum) {
        sum = coord;
      } else {
        sum = arith::AddIOp::create(b, loc, sum, coord);
      }
    }
    offsets.push_back(sum);
  }
  return offsets;
}

} // namespace mlir::iree_compiler
