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
           DeltaFunc deltaFunc, ArrayRef<Chunk> maybeInterestingChunks) {
  SmallVector<Chunk> currentChunks;
  copy_if(maybeInterestingChunks, std::back_inserter(currentChunks),
          [&](Chunk chunk) { return chunk != maybeUninterestingChunk; });

  ChunkManager chunker(currentChunks);
  std::unique_ptr<WorkItem> clonedProgram = root.clone();
  deltaFunc(chunker, *clonedProgram.get());

  if (!oracle.isInteresting(*clonedProgram)) {
    return nullptr;
  }

  return clonedProgram;
};

static bool increaseGranuality() { return false; };

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
  assert(!oracle.isInteresting(root) &&
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

    for (Chunk chunk : maybeInteresting) {
      std::unique_ptr<WorkItem> result = nullptr;
      result = checkChunk(chunk, oracle, root, deltaFunc, maybeInteresting);
      if (!result)
        continue;

      // Removing this chunk is still interesting. Mark this chunk as
      // uninteresting.
      atleastOneNewUninteresting = true;
      reducedProgram = std::move(result);
    }

  } while (!maybeInteresting.empty() &&
           (atleastOneNewUninteresting || increaseGranuality()));
}
