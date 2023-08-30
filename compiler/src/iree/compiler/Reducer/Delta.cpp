// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Delta.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static std::unique_ptr<WorkItem>
checkChunk(Chunk maybeUninterestingChunk, Oracle &oracle, WorkItem &root,
           DeltaFunc deltaFunc, ArrayRef<Chunk> maybeInterestingChunks,
           DenseSet<Chunk> &uninterestingChunks) {
  SmallVector<Chunk> currentChunks;
  copy_if(maybeInterestingChunks, std::back_inserter(currentChunks),
          [&](Chunk chunk) {
            return chunk == maybeUninterestingChunk ||
                   !uninterestingChunks.count(chunk);
          });

  ChunkManager chunker(currentChunks);
  std::unique_ptr<WorkItem> clonedProgram = root.clone();
  deltaFunc(chunker, *clonedProgram.get());

  if (!oracle.isInteresting(*clonedProgram)) {
    return nullptr;
  }

  return clonedProgram;
};

static bool increaseGranuality(SmallVector<Chunk> &chunks) {
  llvm::errs() << "Increasing granularity\n";
  SmallVector<Chunk> newChunks;
  bool anyNewSplit = false;

  for (Chunk c : chunks) {
    if (c.getEnd() - c.getBegin() <= 1) {
      newChunks.push_back(c);
    } else {
      int half = (c.getEnd() + c.getBegin()) / 2;
      newChunks.push_back(Chunk(c.getBegin(), half));
      newChunks.push_back(Chunk(half, c.getEnd()));
      anyNewSplit = true;
    }
  }

  if (anyNewSplit) {
    chunks = newChunks;
    llvm::errs() << "Successfully increased granularity\n";
    llvm::errs() << "New Chunks:\n";
    for (Chunk c : chunks) {
      c.dump();
    }
  }

  return anyNewSplit;
};

void mlir::iree_compiler::runDeltaPass(Oracle &oracle, WorkItem &root,
                                       DeltaFunc deltaFunc, StringRef message) {
  assert(root.verify().succeeded() && "Input module does not verify.");
  llvm::errs() << "=== " << message << " ===\n";

  // Call the delta function with the whole program as the chunk.
  SmallVector<Chunk> chunks = {Chunk(UINT_MAX)};
  ChunkManager chunkManager(chunks);
  deltaFunc(chunkManager, root);
  int numTargets = chunkManager.getCurrentFeatureCount();

  assert(root.verify().succeeded() &&
         "Output module does not verify after counting chunks.");
  assert(oracle.isInteresting(root) &&
         "Output module not interesting after counting chunks.");

  if (!numTargets) {
    llvm::errs() << "\nNothing to reduce\n";
    llvm::errs() << "--------------------------------";
    return;
  }

  SmallVector<Chunk> maybeInteresting = {Chunk(numTargets)};
  std::unique_ptr<WorkItem> reducedProgram;

  bool atleastOneNewUninteresting;
  do {
    atleastOneNewUninteresting = false;
    DenseSet<Chunk> uninterestingChunks;

    for (Chunk chunk : maybeInteresting) {
      std::unique_ptr<WorkItem> result = nullptr;
      result = checkChunk(chunk, oracle, root, deltaFunc, maybeInteresting,
                          uninterestingChunks);
      if (!result)
        continue;

      // Removing this chunk is still interesting. Mark this chunk as
      // uninteresting.
      uninterestingChunks.insert(chunk);
      atleastOneNewUninteresting = true;
      reducedProgram = std::move(result);
    }

    erase_if(maybeInteresting,
             [&](Chunk chunk) { return uninterestingChunks.count(chunk) > 0; });

  } while (!maybeInteresting.empty() && (atleastOneNewUninteresting ||
                                         increaseGranuality(maybeInteresting)));
}
