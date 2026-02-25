// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS
#define IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS

include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapBase.td"

//===----------------------------------------------------------------------===//
// PackMapAttr
//===----------------------------------------------------------------------===//

def IREEMap_PackMapAttr : IREEMap_Attr<"PackMap"> {
  let mnemonic = "pack_map";

  let summary = [{A layout map: a (Shape, Stride) pair mapping coordinates to indices.}];

  let description = [{
    A layout map is a function from coordinates to 1-D indices, defined by a
    `(Shape, Stride)` pair of hierarchical IntTuples.

    #### IntTuple

    An IntTuple is a recursive type: either a single integer or a tuple of
    IntTuples. For example: `4`, `(2, 3)`, `((2, 4), 8)`.

    All values are static i32 constants.

    #### Ordering convention

    This implementation uses **lexicographic (row-major)** ordering, where the
    rightmost mode is the innermost (fastest-varying) dimension. The natural
    stride for shape `(M, N, K)` is the suffix product `(N*K, K, 1)`.

    The layout algebra (coalesce, compose, complement, logical_divide,
    logical_product) all follow this lexicographic convention.

    #### Layout evaluation

    Given a coordinate, a layout computes an index via:

    ```
    layout(coord) = crd2idx(idx2crd(coord, shape), shape, stride)
    ```

    #### Congruence

    Shape and stride must be **congruent**: they have identical hierarchical
    tree structure.

    #### Layout algebra

    PackMap supports a closed algebra:
    - `coalesce`: merge contiguous modes (simplify without changing function)
    - `compose`: functional composition `R(c) = A(B(c))`
    - `complement`: find layout covering indices A doesn't cover
    - `logicalDivide`: factor into tile + rest
    - `logicalProduct`: replicate a pattern
    - `filter`: remove stride-0 and size-1 modes, then coalesce
    - `rightInverse`: find R such that A(R(x)) = x
    - `leftInverse`: find L such that L(A(x)) = x
    - `zippedDivide` / `tiledDivide`: hierarchical logical divide
    - `zippedProduct` / `tiledProduct`: hierarchical logical product

    #### Assembly format

    ```mlir
    #iree_map.pack_map<(8) : (1)>                        // rank-1 contiguous
    #iree_map.pack_map<(4, 8) : (8, 1)>                  // rank-2 row-major
    #iree_map.pack_map<(4, 8) : (1, 4)>                  // rank-2 column-major
    #iree_map.pack_map<((2, 4), 8) : ((16, 1), 4)>       // hierarchical
    ```

    **Important**: PackMap does not coalesce on construction. It stores the
    map exactly as built. Use `coalesce()` explicitly when needed.
  }];

  let parameters = (ins
    "::mlir::Attribute":$shape,
    "::mlir::Attribute":$stride
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // --- Properties ---
    int32_t getRank();
    int32_t getDepth();
    int32_t getSize();
    int32_t getCosize();

    // --- Mode access ---
    ::mlir::Attribute getShapeMode(int32_t i);
    ::mlir::Attribute getStrideMode(int32_t i);

    // --- Evaluation ---
    int32_t evaluate(::llvm::ArrayRef<int32_t> coord);

    // --- Simplification ---
    PackMapAttr coalesce();
    PackMapAttr flatten();

    // --- Algebra ---
    PackMapAttr compose(PackMapAttr rhs);
    PackMapAttr complement(int32_t cotarget);
    PackMapAttr logicalDivide(PackMapAttr tiler);
    PackMapAttr logicalProduct(PackMapAttr tiler);

    // --- Extended algebra ---
    // filter: remove size-1 and stride-0 leaves, then coalesce.
    PackMapAttr filter();
    // rightInverse: find R such that A(R(x)) = x. Requires A to be injective.
    PackMapAttr rightInverse();
    // leftInverse: find L such that L(A(x)) = x.
    PackMapAttr leftInverse();
    // zippedDivide: logicalDivide returning a rank-2 [tiler, rest] layout.
    PackMapAttr zippedDivide(PackMapAttr tiler);
    // tiledDivide: like zippedDivide but tiler mode comes first.
    PackMapAttr tiledDivide(PackMapAttr tiler);
    // zippedProduct: logicalProduct returning a rank-2 [A, rest] layout.
    PackMapAttr zippedProduct(PackMapAttr tiler);
    // tiledProduct: like zippedProduct but tiler mode comes first.
    PackMapAttr tiledProduct(PackMapAttr tiler);

    // --- Mode operations ---
    PackMapAttr permute(::llvm::ArrayRef<int64_t> perm);
    PackMapAttr project(::llvm::ArrayRef<bool> droppedDims);

    // --- Factory ---
    // makeIdentity: create the row-major (lexicographic) identity layout for a
    // flat shape. For shape (M, N, K) produces (M, N, K) : (N*K, K, 1).
    static PackMapAttr makeIdentity(::mlir::MLIRContext *ctx,
                                    ::llvm::ArrayRef<int64_t> shape);
  }];
}

//===----------------------------------------------------------------------===//
// PackLayoutAttr
//===----------------------------------------------------------------------===//

def IREEMap_PackLayoutAttr : IREEMap_Attr<"PackLayout"> {
  let mnemonic = "pack_layout";

  let summary = [{A layout wrapping a PackMap that always by-mode-coalesces.}];

  let description = [{
    PackLayout wraps a PackMap and always stores it in by-mode-coalesced form.
    Each top-level mode is independently coalesced (adjacent leaves with
    compatible strides are merged), but the top-level rank is preserved.

    This attribute implements VectorLayoutInterface as an ExternalInterface.

    #### Assembly format

    ```mlir
    #iree_map.pack_layout<(4, 8) : (8, 1)>
    ```

    The printed form shows the coalesced shape and stride. Construction from
    un-coalesced input will automatically coalesce each mode.
  }];

  let parameters = (ins
    "PackMapAttr":$map
  );

  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilder<(ins "PackMapAttr":$map), [{
      // By-mode-coalesce: coalesce each top-level mode independently.
      ::mlir::Attribute shape = map.getShape();
      ::mlir::Attribute stride = map.getStride();
      if (isLeaf(shape))
        return $_get(context, map);
      auto shapeArr = ::mlir::cast<::mlir::ArrayAttr>(shape);
      auto strideArr = ::mlir::cast<::mlir::ArrayAttr>(stride);
      ::llvm::SmallVector<::mlir::Attribute> newShapes, newStrides;
      for (unsigned i = 0; i < shapeArr.size(); ++i) {
        auto [cs, cd] = coalesceMode(context, shapeArr[i], strideArr[i]);
        newShapes.push_back(cs);
        newStrides.push_back(cd);
      }
      auto coalescedMap = PackMapAttr::get(context,
                                           makeTuple(context, newShapes),
                                           makeTuple(context, newStrides));
      return $_get(context, coalescedMap);
    }]>,
    AttrBuilder<(ins "::mlir::Attribute":$shape, "::mlir::Attribute":$stride), [{
      auto map = PackMapAttr::get(context, shape, stride);
      return PackLayoutAttr::get(context, map);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Convenience accessors delegating to the underlying map.
    int32_t getRank() { return getMap().getRank(); }
    ::mlir::Attribute getShape() { return getMap().getShape(); }
    ::mlir::Attribute getStride() { return getMap().getStride(); }
    ::mlir::Attribute getShapeMode(int32_t i) { return getMap().getShapeMode(i); }
    ::mlir::Attribute getStrideMode(int32_t i) { return getMap().getStrideMode(i); }

    // Mode operations that return new PackLayoutAttrs.
    PackLayoutAttr permute(::llvm::ArrayRef<int64_t> perm);
    PackLayoutAttr project(::llvm::ArrayRef<bool> droppedDims);

    // Coalesce the underlying map (full coalesce, not by-mode).
    PackMapAttr coalesce() { return getMap().coalesce(); }
  }];
}

#endif // IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS
