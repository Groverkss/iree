// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Map;

//===----------------------------------------------------------------------===//
// IntTuple parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parse an IntTuple: either a bare integer or `(` IntTuple (`,` IntTuple)*
/// `)`.
static FailureOr<Attribute> parseIntTuple(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Attribute> elements;
    auto result = parseIntTuple(parser);
    if (failed(result)) {
      return failure();
    }
    elements.push_back(*result);
    while (succeeded(parser.parseOptionalComma())) {
      result = parseIntTuple(parser);
      if (failed(result)) {
        return failure();
      }
      elements.push_back(*result);
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
    return makeTuple(ctx, elements);
  }

  int64_t val64;
  if (failed(parser.parseInteger(val64))) {
    return failure();
  }
  return makeLeaf(ctx, static_cast<int32_t>(val64));
}

/// Print an IntTuple.
static void printIntTuple(AsmPrinter &printer, Attribute attr) {
  if (isLeaf(attr)) {
    printer << getLeafValue(attr);
    return;
  }
  auto arr = cast<ArrayAttr>(attr);
  printer << "(";
  llvm::interleaveComma(arr, printer,
                        [&](Attribute elem) { printIntTuple(printer, elem); });
  printer << ")";
}

//===----------------------------------------------------------------------===//
// PackMapAttr — parsing/printing
//===----------------------------------------------------------------------===//

Attribute PackMapAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  auto shape = parseIntTuple(parser);
  if (failed(shape)) {
    return {};
  }

  if (failed(parser.parseColon())) {
    return {};
  }

  auto stride = parseIntTuple(parser);
  if (failed(stride)) {
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }

  return PackMapAttr::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), *shape, *stride);
}

void PackMapAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntTuple(printer, getShape());
  printer << " : ";
  printIntTuple(printer, getStride());
  printer << ">";
}

LogicalResult PackMapAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Attribute shape, Attribute stride) {
  if (!isIntTuple(shape)) {
    return emitError() << "shape must be a valid IntTuple";
  }
  if (!isIntTuple(stride)) {
    return emitError() << "stride must be a valid IntTuple";
  }

  auto checkPositive = [&](Attribute attr, StringRef name) -> LogicalResult {
    SmallVector<int32_t> leaves = getLeaves(attr);
    for (int32_t v : leaves) {
      if (v <= 0) {
        return emitError() << name << " leaf values must be positive, got "
                           << v;
      }
    }
    return success();
  };
  if (failed(checkPositive(shape, "shape"))) {
    return failure();
  }

  SmallVector<int32_t> strideLeaves = getLeaves(stride);
  for (int32_t v : strideLeaves) {
    if (v < 0) {
      return emitError() << "stride leaf values must be non-negative, got "
                         << v;
    }
  }

  if (!isCongruent(shape, stride)) {
    return emitError() << "shape and stride must be congruent (identical tree "
                          "structure)";
  }

  if (isLeaf(shape)) {
    return emitError()
           << "top-level shape must be a tuple (use (N) : (S) for rank-1)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PackMapAttr — property methods
//===----------------------------------------------------------------------===//

int32_t PackMapAttr::getRank() { return Map::getRank(getShape()); }

int32_t PackMapAttr::getDepth() { return Map::getDepth(getShape()); }

int32_t PackMapAttr::getSize() { return Map::getSize(getShape()); }

int32_t PackMapAttr::getCosize() {
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());
  int32_t result = 0;
  for (auto [s, d] : llvm::zip(shapes, strides)) {
    result += (s - 1) * d;
  }
  return result + 1;
}

Attribute PackMapAttr::getShapeMode(int32_t i) {
  return Map::getElement(getShape(), i);
}

Attribute PackMapAttr::getStrideMode(int32_t i) {
  return Map::getElement(getStride(), i);
}

//===----------------------------------------------------------------------===//
// PackMapAttr — evaluation
//===----------------------------------------------------------------------===//

int32_t PackMapAttr::evaluate(ArrayRef<int32_t> coord) {
  SmallVector<int32_t> strides = getLeaves(getStride());
  SmallVector<int32_t> shapes = getLeaves(getShape());
  assert(shapes.size() == strides.size());

  SmallVector<int32_t> natCoord;
  if (coord.size() == 1 && shapes.size() > 1) {
    natCoord = idx2crd(coord[0], getShape());
  } else if (coord.size() == shapes.size()) {
    natCoord.assign(coord.begin(), coord.end());
  } else {
    int32_t flatIdx = 0;
    SmallVector<int32_t> modeShapes;
    for (int32_t i = 0; i < getRank(); ++i) {
      modeShapes.push_back(Map::getSize(getShapeMode(i)));
    }
    for (int32_t i = coord.size() - 1; i >= 0; --i) {
      flatIdx = flatIdx * modeShapes[i] + coord[i];
    }
    natCoord = idx2crd(flatIdx, getShape());
  }

  return crd2idx(natCoord, getShape(), getStride());
}

//===----------------------------------------------------------------------===//
// PackMapAttr — simplification
//===----------------------------------------------------------------------===//

PackMapAttr PackMapAttr::flatten() {
  MLIRContext *ctx = getContext();
  return PackMapAttr::get(ctx, Map::flatten(ctx, getShape()),
                          Map::flatten(ctx, getStride()));
}

// Port of _pycute/layout.py:164 (coalesce)
PackMapAttr PackMapAttr::coalesce() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());
  assert(shapes.size() == strides.size());

  if (shapes.empty()) {
    return *this;
  }

  SmallVector<std::pair<int32_t, int32_t>> result;
  result.push_back({shapes.back(), strides.back()});

  for (int i = static_cast<int>(shapes.size()) - 2; i >= 0; --i) {
    int32_t si = shapes[i];
    int32_t di = strides[i];
    auto &[accShape, accStride] = result.back();

    if (si == 1) {
      continue;
    }
    if (accShape == 1) {
      accShape = si;
      accStride = di;
      continue;
    }
    if (accShape * accStride == di) {
      accShape = si * accShape;
      continue;
    }
    result.push_back({si, di});
  }

  std::reverse(result.begin(), result.end());

  SmallVector<Attribute> newShape, newStride;
  for (auto [s, d] : result) {
    newShape.push_back(makeLeaf(ctx, s));
    newStride.push_back(makeLeaf(ctx, s == 1 ? 0 : d));
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

//===----------------------------------------------------------------------===//
// PackMapAttr — algebra
//===----------------------------------------------------------------------===//

// Port of _pycute/layout.py:235 (composition)
PackMapAttr PackMapAttr::compose(PackMapAttr rhs) {
  MLIRContext *ctx = getContext();
  PackMapAttr lhs = this->coalesce();
  SmallVector<int32_t> lhsShapes = getLeaves(lhs.getShape());
  SmallVector<int32_t> lhsStrides = getLeaves(lhs.getStride());

  SmallVector<int32_t> rhsShapes = getLeaves(rhs.getShape());
  SmallVector<int32_t> rhsStrides = getLeaves(rhs.getStride());

  SmallVector<Attribute> resultShapes, resultStrides;
  int n = static_cast<int>(lhsShapes.size());

  for (auto [rhsS, rhsD] : llvm::zip(rhsShapes, rhsStrides)) {
    if (rhsD == 0) {
      resultShapes.push_back(makeLeaf(ctx, rhsS));
      resultStrides.push_back(makeLeaf(ctx, 0));
      continue;
    }

    if (n == 1) {
      resultShapes.push_back(makeLeaf(ctx, rhsS));
      resultStrides.push_back(makeLeaf(ctx, rhsD * lhsStrides[0]));
      continue;
    }

    SmallVector<Attribute> modeShapes, modeStrides;
    int32_t restShape = rhsS;
    int32_t restStride = rhsD;

    for (int j = n - 1; j >= 1; --j) {
      int32_t currShape = lhsShapes[j];
      int32_t currStride = lhsStrides[j];

      assert((currShape % restStride == 0 || restStride % currShape == 0) &&
             "Stride Divisibility Condition");

      int32_t newShape =
          std::min(std::max(1, currShape / restStride), restShape);

      if (newShape != 1) {
        modeShapes.push_back(makeLeaf(ctx, newShape));
        modeStrides.push_back(makeLeaf(ctx, restStride * currStride));
      }

      restShape = restShape / newShape;
      restStride = ceilDiv(restStride, currShape);
    }

    if (restShape != 1 || modeShapes.empty()) {
      modeShapes.push_back(makeLeaf(ctx, restShape));
      modeStrides.push_back(makeLeaf(ctx, restStride * lhsStrides[0]));
    }

    std::reverse(modeShapes.begin(), modeShapes.end());
    std::reverse(modeStrides.begin(), modeStrides.end());

    if (modeShapes.size() == 1) {
      resultShapes.push_back(modeShapes[0]);
      resultStrides.push_back(modeStrides[0]);
    } else {
      resultShapes.push_back(makeTuple(ctx, modeShapes));
      resultStrides.push_back(makeTuple(ctx, modeStrides));
    }
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, resultShapes),
                          makeTuple(ctx, resultStrides));
}

// Port of _pycute/layout.py:303 (complement)
PackMapAttr PackMapAttr::complement(int32_t cotarget) {
  MLIRContext *ctx = getContext();

  auto [filtShape, filtStride] = filterZeros(ctx, getShape(), getStride());
  SmallVector<int32_t> shapes = getLeaves(filtShape);
  SmallVector<int32_t> strides = getLeaves(filtStride);

  SmallVector<std::pair<int32_t, int32_t>> modes;
  for (auto [s, d] : llvm::zip(shapes, strides)) {
    modes.push_back({s, d});
  }
  llvm::sort(modes, [](auto &a, auto &b) { return a.second < b.second; });

  SmallVector<Attribute> compShape, compStride;
  int32_t accumulated = 1;
  for (auto [s, d] : modes) {
    int32_t gap = d / accumulated;
    if (gap > 1) {
      compShape.push_back(makeLeaf(ctx, gap));
      compStride.push_back(makeLeaf(ctx, accumulated));
    }
    accumulated = d * s;
  }

  {
    int32_t remaining = ceilDiv(cotarget, accumulated);
    compShape.push_back(makeLeaf(ctx, remaining));
    compStride.push_back(makeLeaf(ctx, accumulated));
  }

  if (compShape.empty()) {
    compShape.push_back(makeLeaf(ctx, 1));
    compStride.push_back(makeLeaf(ctx, 0));
  }

  std::reverse(compShape.begin(), compShape.end());
  std::reverse(compStride.begin(), compStride.end());

  auto result = PackMapAttr::get(ctx, makeTuple(ctx, compShape),
                                 makeTuple(ctx, compStride));
  return result.coalesce();
}

PackMapAttr PackMapAttr::logicalDivide(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();

  PackMapAttr coal = this->coalesce();
  PackMapAttr tilerComp = tiler.complement(coal.getSize());

  PackMapAttr innerResult = coal.compose(tiler);
  PackMapAttr outerResult = coal.compose(tilerComp);

  SmallVector<Attribute> resShape = {innerResult.getShape(),
                                     outerResult.getShape()};
  SmallVector<Attribute> resStride = {innerResult.getStride(),
                                      outerResult.getStride()};
  return PackMapAttr::get(ctx, makeTuple(ctx, resShape),
                          makeTuple(ctx, resStride));
}

PackMapAttr PackMapAttr::permute(ArrayRef<int64_t> perm) {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (int64_t i : perm) {
    newShape.push_back(getShapeMode(i));
    newStride.push_back(getStrideMode(i));
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

PackMapAttr PackMapAttr::project(ArrayRef<bool> droppedDims) {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (size_t i = 0; i < droppedDims.size(); ++i) {
    if (!droppedDims[i]) {
      newShape.push_back(getShapeMode(i));
      newStride.push_back(getStrideMode(i));
    }
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

PackMapAttr PackMapAttr::logicalProduct(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();

  int32_t target = getSize() * tiler.getCosize();
  PackMapAttr comp = this->complement(target);
  PackMapAttr mode1 = comp.compose(tiler);

  SmallVector<Attribute> combinedShape = {getShape(), mode1.getShape()};
  SmallVector<Attribute> combinedStride = {getStride(), mode1.getStride()};
  return PackMapAttr::get(ctx, makeTuple(ctx, combinedShape),
                          makeTuple(ctx, combinedStride));
}

// Port of _pycute/layout.py:207 (filter)
PackMapAttr PackMapAttr::filter() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());

  SmallVector<Attribute> filtShape, filtStride;
  for (auto [s, d] : llvm::zip(shapes, strides)) {
    if (s != 1 && d != 0) {
      filtShape.push_back(makeLeaf(ctx, s));
      filtStride.push_back(makeLeaf(ctx, d));
    }
  }

  if (filtShape.empty()) {
    filtShape.push_back(makeLeaf(ctx, 1));
    filtStride.push_back(makeLeaf(ctx, 0));
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, filtShape),
                          makeTuple(ctx, filtStride))
      .coalesce();
}

// Port of _pycute/layout.py:336 (right_inverse)
PackMapAttr PackMapAttr::rightInverse() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());

  int n = static_cast<int>(shapes.size());
  SmallVector<int32_t> rStrides(n);
  {
    int32_t acc = 1;
    for (int i = n - 1; i >= 0; --i) {
      rStrides[i] = acc;
      acc *= shapes[i];
    }
  }

  SmallVector<std::tuple<int32_t, int32_t, int32_t>> sorted;
  for (int i = 0; i < n; ++i) {
    sorted.push_back({strides[i], shapes[i], rStrides[i]});
  }
  llvm::sort(sorted,
             [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });

  SmallVector<Attribute> resShapes, resStrides;
  int32_t currentIdx = 1;
  for (auto [stride, shape, rStride] : sorted) {
    if (shape == 1) {
      continue;
    }
    if (currentIdx != stride) {
      break;
    }
    resShapes.push_back(makeLeaf(ctx, shape));
    resStrides.push_back(makeLeaf(ctx, rStride));
    currentIdx = shape * stride;
  }

  if (resShapes.empty()) {
    resShapes.push_back(makeLeaf(ctx, 1));
    resStrides.push_back(makeLeaf(ctx, 0));
  }

  std::reverse(resShapes.begin(), resShapes.end());
  std::reverse(resStrides.begin(), resStrides.end());

  return PackMapAttr::get(ctx, makeTuple(ctx, resShapes),
                          makeTuple(ctx, resStrides))
      .coalesce();
}

// Port of _pycute/layout.py:365 (left_inverse)
PackMapAttr PackMapAttr::leftInverse() {
  MLIRContext *ctx = getContext();
  PackMapAttr comp = this->complement(getSize());

  SmallVector<Attribute> combinedShape = {comp.getShape(), getShape()};
  SmallVector<Attribute> combinedStride = {comp.getStride(), getStride()};
  PackMapAttr combined = PackMapAttr::get(ctx, makeTuple(ctx, combinedShape),
                                          makeTuple(ctx, combinedStride));
  return combined.rightInverse();
}

//===----------------------------------------------------------------------===//
// PackMapAttr — zipped/tiled divide and product
//===----------------------------------------------------------------------===//

// Port of _pycute/layout.py:459 (zipped_divide)
PackMapAttr PackMapAttr::zippedDivide(PackMapAttr tiler) {
  return this->logicalDivide(tiler);
}

// Port of _pycute/layout.py:464 (tiled_divide)
PackMapAttr PackMapAttr::tiledDivide(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();
  PackMapAttr divided = zippedDivide(tiler);
  SmallVector<Attribute> newShape = {divided.getShapeMode(0)};
  SmallVector<Attribute> newStride = {divided.getStrideMode(0)};
  Attribute restShape = divided.getShapeMode(1);
  Attribute restStride = divided.getStrideMode(1);
  if (isLeaf(restShape)) {
    newShape.push_back(restShape);
    newStride.push_back(restStride);
  } else {
    for (Attribute s : cast<ArrayAttr>(restShape)) {
      newShape.push_back(s);
    }
    for (Attribute d : cast<ArrayAttr>(restStride)) {
      newStride.push_back(d);
    }
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

// Port of _pycute/layout.py:470 (zipped_product)
PackMapAttr PackMapAttr::zippedProduct(PackMapAttr tiler) {
  return logicalProduct(tiler);
}

// Port of _pycute/layout.py:474 (tiled_product)
PackMapAttr PackMapAttr::tiledProduct(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();
  PackMapAttr zipped = zippedProduct(tiler);
  SmallVector<Attribute> newShape = {zipped.getShapeMode(0)};
  SmallVector<Attribute> newStride = {zipped.getStrideMode(0)};
  Attribute restShape = zipped.getShapeMode(1);
  Attribute restStride = zipped.getStrideMode(1);
  if (isLeaf(restShape)) {
    newShape.push_back(restShape);
    newStride.push_back(restStride);
  } else {
    for (Attribute s : cast<ArrayAttr>(restShape)) {
      newShape.push_back(s);
    }
    for (Attribute d : cast<ArrayAttr>(restStride)) {
      newStride.push_back(d);
    }
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

//===----------------------------------------------------------------------===//
// PackMapAttr — factory
//===----------------------------------------------------------------------===//

PackMapAttr PackMapAttr::makeIdentity(MLIRContext *ctx,
                                      ArrayRef<int64_t> shape) {
  SmallVector<Attribute> leaves;
  for (int64_t s : shape) {
    leaves.push_back(makeLeaf(ctx, static_cast<int32_t>(s)));
  }
  Attribute shapeAttr = makeTuple(ctx, leaves);
  return PackMapAttr::get(ctx, shapeAttr, suffixProduct(ctx, shapeAttr));
}

//===----------------------------------------------------------------------===//
// PackLayoutAttr — parsing/printing
//===----------------------------------------------------------------------===//

Attribute PackLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  auto shape = parseIntTuple(parser);
  if (failed(shape)) {
    return {};
  }

  if (failed(parser.parseColon())) {
    return {};
  }

  auto stride = parseIntTuple(parser);
  if (failed(stride)) {
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }

  return PackLayoutAttr::get(parser.getContext(), *shape, *stride);
}

void PackLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntTuple(printer, getMap().getShape());
  printer << " : ";
  printIntTuple(printer, getMap().getStride());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// PackLayoutAttr — mode operations
//===----------------------------------------------------------------------===//

PackLayoutAttr PackLayoutAttr::permute(ArrayRef<int64_t> perm) {
  return PackLayoutAttr::get(getContext(), getMap().permute(perm));
}

PackLayoutAttr PackLayoutAttr::project(ArrayRef<bool> droppedDims) {
  return PackLayoutAttr::get(getContext(), getMap().project(droppedDims));
}

//===----------------------------------------------------------------------===//
// Dialect attribute registration
//===----------------------------------------------------------------------===//

void IREEMapDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen generated definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.cpp.inc"
