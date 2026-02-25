// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE
#define IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// IREE Map Dialect
//===----------------------------------------------------------------------===//

def IREEMap_Dialect : Dialect {
  let name = "iree_map";
  let cppNamespace = "::mlir::iree_compiler::IREE::Map";

  let summary = [{
    A dialect for layout algebra used in GPU code generation.
  }];
  let description = [{
    This dialect provides layout attributes for describing hierarchical
    thread-value mappings in GPU kernels. Layouts are (Shape, Stride) pairs
    where both are hierarchical IntTuples (recursive trees of integers).
    They support a closed algebra: composition, complement, logical division,
    and logical product.

    The primary use case is describing how MMA and copy instructions
    distribute data across GPU threads via Thread-Value (TV) layouts.
  }];

  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//

class IREEMap_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREEMap_Dialect, name, traits>;

#endif // IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE
