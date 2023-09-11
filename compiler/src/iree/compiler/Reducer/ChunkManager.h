// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_CHUNK_MANAGER_H
#define IREE_COMPILER_REDUCER_CHUNK_MANAGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

/// Represents the range: [begin, end).
class Chunk {
public:
  Chunk(unsigned begin, unsigned end) : begin(begin), end(end){};
  Chunk(unsigned numTargets) : Chunk(0, numTargets){};

  unsigned getBegin() const { return begin; }
  unsigned getEnd() const { return end; }

  bool contains(unsigned index) const;

  friend bool operator==(const Chunk &C1, const Chunk &C2);
  friend bool operator!=(const Chunk &C1, const Chunk &C2);
  friend bool operator<(const Chunk &C1, const Chunk &C2);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  unsigned begin;
  unsigned end;
};

/// Interface for a delta pass to query into chunks.
class ChunkManager {
public:
  explicit ChunkManager(ArrayRef<Chunk> chunksToKeep)
      : chunksToKeep(chunksToKeep){};

  /// Should be called for each feature being processed. Returns true if the
  /// feature should be preserved.
  bool shouldFeatureBeKept();

  unsigned getCurrentFeatureCount() { return featureIndex; };

private:
  // The index of the feature being processed.
  unsigned featureIndex = 0;

  ArrayRef<Chunk> chunksToKeep;
};

} // namespace iree_compiler
} // namespace mlir

namespace llvm {
using Chunk = mlir::iree_compiler::Chunk;

template <>
struct DenseMapInfo<Chunk> {
  static inline Chunk getEmptyKey() {
    return Chunk(DenseMapInfo<unsigned>::getEmptyKey(),
                 DenseMapInfo<unsigned>::getEmptyKey());
  }

  static inline Chunk getTombstoneKey() {
    return Chunk(DenseMapInfo<unsigned>::getTombstoneKey(),
                 DenseMapInfo<unsigned>::getTombstoneKey());
  }

  static unsigned getHashValue(const Chunk &val) {
    std::pair<unsigned, unsigned> pair(val.getBegin(), val.getEnd());
    return DenseMapInfo<std::pair<unsigned, unsigned>>::getHashValue(pair);
  }

  static bool isEqual(const Chunk &LHS, const Chunk &RHS) { return LHS == RHS; }
};
} // namespace llvm

#endif // IREE_COMPILER_REDUCER_CHUNK_MANAGER_H