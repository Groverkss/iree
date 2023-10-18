// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;

#define DEBUG_TYPE "llvm-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {

static bool isContiguousStore(Operation *write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      LDBG("--not a contiguous store op: " << *write);
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  LDBG("--not a store op: " << write->getName().getStringRef());
  return false;
}

static bool isContiguousRead(Operation *read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      LDBG("--not a contiguous load op: " << *read);
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  LDBG("--not a load op: " << read->getName().getStringRef());
  return false;
}

static Value getMemrefOperand(Operation *op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

struct MaskResult {
  vector::CreateMaskOp maskOp;
  vector::ExtractOp maybeExtractOp;
};
static MaskResult getMask(Operation *op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask())
    return MaskResult{};
  vector::ExtractOp maybeExtractOp =
      transferRead.getMask().getDefiningOp<vector::ExtractOp>();
  auto maskOp =
      maybeExtractOp
          ? maybeExtractOp.getVector().getDefiningOp<vector::CreateMaskOp>()
          : transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  if (maybeExtractOp) {
    if (maybeExtractOp.getPosition().size() + 1 !=
        llvm::cast<VectorType>(maskOp->getResultTypes().front()).getRank()) {
      LDBG("----mask through extract unexpected position size -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    if (maybeExtractOp.getPosition().size() != 1) {
      LDBG("----only mask through 2-D -> 1-D extract supported atm -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    LDBG("----mask through extract: " << maybeExtractOp);
  }
  return MaskResult{maskOp, maybeExtractOp};
}

static Value getMaskValue(RewriterBase &rewriter, Operation *op) {
  MaskResult maskResult = getMask(op);
  if (!maskResult.maskOp)
    return Value();
  Value count = maskResult.maskOp->getOperands().back();
  vector::ExtractOp maybeExtractOp = maskResult.maybeExtractOp;
  if (maybeExtractOp) {
    assert(maybeExtractOp.getPosition().size() == 1 && "expected single pos");
    int64_t sliceNum = maybeExtractOp.getPosition()[0];
    // TODO: to support >2-D mask + extract, and all the cmp.
    Location loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        rewriter.create<arith::ConstantIndexOp>(loc, sliceNum),
        maskResult.maskOp->getOperands().front());
    count = rewriter.create<arith::SelectOp>(loc, cmp, count, zero);
  }
  return count;
}

static Value getValueStored(Operation *writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation *op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

/// Return `true` if the conversion to async copy is legal.
static bool resultsInSupportedAsyncCopy(MemRefType memrefType,
                                        Operation::operand_range indices,
                                        VectorType vecType) {
  constexpr int64_t kSupportedCpAsyncAlignmentsInBytes[3] = {4, 8, 16};
  // Condition 1: the vectory rank must be supported.
  if (vecType.hasRank() != 1) {
    LDBG("----> cp.async failed, not a 1-D vector: " << vecType);
    return false;
  }

  // Condition 2: the copy size must be supported.
  bool supportedCopySize = false;
  int64_t numElements = vecType.getNumElements();
  Type elementType = vecType.getElementType();
  for (int64_t alignmentInBytes : kSupportedCpAsyncAlignmentsInBytes) {
    if (alignmentInBytes * 8 ==
        numElements * elementType.getIntOrFloatBitWidth()) {
      supportedCopySize = true;
      break;
    }
  }
  if (!supportedCopySize) {
    LDBG("----> cp.async alignment failed, "
         << numElements << " elts * " << elementType.getIntOrFloatBitWidth()
         << "b/elem = " << numElements * elementType.getIntOrFloatBitWidth()
         << "b is not supported by cp.async");
    return false;
  }

  // TODO: Condition 3: the alignments must be supported. For cp.async the
  // NVIDIA doc (section 6.4.1) says: "The address must be naturally aligned to
  // a multiple of the access size. If an address is not properly aligned, the
  // resulting behavior is undefined.".
  return true;
}

void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync) {
  LDBG("Start asyncGroups: useMMASync=" << useMMASync);
  llvm::SmallSetVector<Operation *, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation *writeOp) {
    if (!isContiguousStore(writeOp))
      return WalkResult::advance();
    LDBG("--candidate writeOp: " << *writeOp);
    Value vectorVal = getValueStored(writeOp);
    if (llvm::cast<VectorType>(vectorVal.getType()).getRank() != 1) {
      LDBG("----writeOp is not an inbounds 1-D minor identity -> Skip");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(memrefOperand.getType()))) {
      LDBG("----address space is not workgroup -> Skip");
      return WalkResult::advance();
    }
    Operation *readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LDBG("----no contiguous readOp defining the writeOp -> Skip");
      return WalkResult::advance();
    }

    LDBG("--candidate readOp: " << *readOp);
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (transferRead.getMask()) {
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LDBG("----read padding value is not 0.f -> Skip");
          return WalkResult::advance();
        }
        auto maskResult = getMask(transferRead);
        if (!maskResult.maskOp) {
          LDBG("----read mask is not a vector.create_mask op -> Skip: "
               << transferRead.getMask());
          return WalkResult::advance();
        }
      }
    }

    // Check whether both accesses are supported before we emit: this is
    // necessary to ensure the correctness of DeviceAsyncCopyOp.
    VectorType vecType = llvm::cast<VectorType>(vectorVal.getType());
    Value storeBase = getMemrefOperand(writeOp);
    Value loadBase = getMemrefOperand(readOp);
    if (!resultsInSupportedAsyncCopy(cast<MemRefType>(loadBase.getType()),
                                     getIndices(readOp), vecType) ||
        !resultsInSupportedAsyncCopy(cast<MemRefType>(storeBase.getType()),
                                     getIndices(writeOp), vecType))
      return WalkResult::advance();

    LDBG("--writeOp can be made async -> SUCCESS");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation *> group;
    Operation *writeOp = *copyToSharedMem.begin();
    LDBG("--START a group from: " << *writeOp);
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation *nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation *readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                llvm::cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      LDBG("----> STOP accumulating into group due to: " << *nextNode);
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    for (Operation *writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      auto vectorType = llvm::cast<VectorType>(vectorVal.getType());
      int64_t numElements = vectorType.getNumElements();
      Operation *readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMaskValue(rewriter, readOp);
      auto dstMemref = llvm::cast<MemRefType>(storeBase.getType());
      int64_t sizeInBytes =
          (dstMemref.getElementTypeBitWidth() * numElements) / 8;
      UnitAttr bypassL1 =
          useMMASync && sizeInBytes == 16 ? rewriter.getUnitAttr() : UnitAttr();
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(numElements), mask,
          /*bypassL1=*/bypassL1);
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation *writeOp : group)
      rewriter.eraseOp(writeOp);
  }
}

void reorderTranspose(RewriterBase &rewriter, func::FuncOp funcOp) {
  SmallVector<vector::TransposeOp> transposeOps;
  funcOp.walk([&](Operation *op) {
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
      Operation *definingOp = transposeOp.getVector().getDefiningOp();
      if (OpTrait::hasElementwiseMappableTraits(definingOp)) {
        transposeOps.push_back(transposeOp);
      }
    }
    return WalkResult::advance();
  });

  for (auto transposeOp : transposeOps) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation *op = transposeOp.getVector().getDefiningOp();
    rewriter.setInsertionPoint(op);
    SmallVector<int64_t> perm;
    transposeOp.getTransp(perm);
    SmallVector<Value> transposedOperands;
    for (auto operand : op->getOperands()) {
      Value transposed =
          rewriter.create<vector::TransposeOp>(op->getLoc(), operand, perm);
      transposedOperands.push_back(transposed);
    }
    SmallVector<Type> resultTypes{transposedOperands.front().getType()};
    Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        transposedOperands, resultTypes, op->getAttrs());
    rewriter.replaceAllUsesWith(transposeOp.getResult(), newOp->getResult(0));
  }
}

/// Insert barriers and wait operations if there are allocs of a different alias
/// group before the given alloc.
static void addBarrier(func::FuncOp funcOp, Operation *alloc,
                       ArrayRef<Operation *> aliasGroup) {
  Block *entryBlock = &(*funcOp.getBlocks().begin());
  bool needBarrier = false;
  if (alloc->getBlock() != entryBlock) {
    needBarrier = true;
  } else {
    for (Operation &op : entryBlock->getOperations()) {
      if (&op == alloc)
        break;
      if (op.getNumRegions() != 0) {
        needBarrier = true;
        break;
      }
      if (isa<memref::AllocOp>(&op) && !llvm::is_contained(aliasGroup, &op)) {
        needBarrier = true;
        break;
      }
    }
  }
  if (!needBarrier)
    return;
  OpBuilder builder(alloc);
  // TODO: make it a option if needed.
  bool hasAsyncCopies = false;
  if (hasAsyncCopies) {
    Value groupToken = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        SmallVector<Value>());
    builder.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                             builder.getI32IntegerAttr(0));
  }
  builder.create<gpu::BarrierOp>(alloc->getLoc());
}

// /// Return all operations within `parentOp` that read from or write to
// /// `shmMemRef`.
// static linalg::GenericOp getShmWriteOps(Operation *parentOp) {
//   linalg::GenericOp copy;
//   parentOp->walk([&](linalg::GenericOp copyOp) {
//     // Is input in shared mem.
//     if (copyOp.getNumDpsInputs() != 1) {
//       return;
//     }
//     auto in = copyOp.getDpsInputOperand(0)->get();
//     auto inMemref = dyn_cast<MemRefType>(in.getType());
//     if (!inMemref) {
//       return;
//     }
//
//     auto inSpace =
//     dyn_cast<gpu::AddressSpaceAttr>(inMemref.getMemorySpace()); if (!inSpace)
//     {
//       return;
//     }
//     if (inSpace.getValue() != gpu::AddressSpace::Workgroup) {
//       return;
//     }
//
//     // Is output in global mem.
//     if (copyOp.getNumDpsInits() != 1) {
//       return;
//     }
//     auto out = copyOp.getDpsInitOperand(0)->get();
//     auto outMemref = dyn_cast<MemRefType>(out.getType());
//     if (!outMemref) {
//       return;
//     }
//
//     auto outSpace =
//         dyn_cast<IREE::HAL::DescriptorTypeAttr>(outMemref.getMemorySpace());
//     if (!outSpace) {
//       return;
//     }
//     if (outSpace.getValue() != IREE::HAL::DescriptorType::StorageBuffer) {
//       return;
//     }
//
//     copy = copyOp;
//   });
//
//   return copy;
// }

// void tileLDSForGlobalWrite(func::FuncOp funcOp) {
//   // Find Shared --> Global memory ops.
//   Operation *op = getShmWriteOps(funcOp);
//
//   OpBuilder builder(funcOp);
//   IRRewriter rewriter(builder);
//
//   auto tilingInt = dyn_cast<TilingInterface>(op);
//   (void)tilingInt;
//   if (!tilingInt) {
//     return;
//   }
//
//   scf::SCFTilingOptions options;
//   options.setTileSizes({16, 16});
//   auto maybeScfFor = scf::tileUsingSCFForOp(rewriter, tilingInt, options);
//
//   if (failed(maybeScfFor))
//     return;
//
//   auto tilingRes = maybeScfFor.value();
//   rewriter.replaceOp(op, tilingRes.tiledOps);
//   funcOp.dump();
//}

void tileBigBuffers(func::FuncOp funcOp) {
  funcOp.walk([](memref::AllocOp alloc) {
    if (IREE::Util::getRoundedPhysicalStorageSize(alloc.getMemref().getType()) >
        32768) {

      auto memrefType = alloc.getMemref().getType();
      auto shapes = memrefType.getShape();
      assert(shapes.size() == 2);

      unsigned tileX = shapes[0] / 4;
      unsigned tileY = shapes[1] / 4;

      // Create a tiled buffer allocation.
      OpBuilder builder(alloc);
      IRRewriter rewriter(builder);
      auto newAlloc = builder.create<memref::AllocOp>(
          alloc.getLoc(),
          MemRefType::get({tileX, tileY}, memrefType.getElementType(),
                          memrefType.getLayout(), memrefType.getMemorySpace()));

      newAlloc.dump();

      // Materialize the uses of the original alloc with new allocs.
      auto users = alloc->getUsers();

      vector::TransferWriteOp vgprToLDS;
      linalg::GenericOp ldsToGlobal;

      for (auto user : users) {
        if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(user)) {
          vgprToLDS = transferWrite;
        } else if (auto linalgOp = dyn_cast<linalg::GenericOp>(user)) {
          ldsToGlobal = linalgOp;
        }
      }

      auto tilingInt = dyn_cast<TilingInterface>(vgprToLDS);
      scf::SCFTilingOptions options;
    }
  });
}

void packSharedMemoryAlloc(func::FuncOp funcOp) {
  tileBigBuffers(funcOp);
  return;

  DominanceInfo dominators(funcOp);
  SmallVector<Operation *> allocs;
  funcOp.walk([&](memref::AllocOp alloc) {
    if (hasSharedMemoryAddressSpace(alloc.getType())) {
      allocs.push_back(alloc);
    }
  });
  // First sink the alloc as low as possible in the CFG.
  sinkOpsInCFG(allocs, dominators);
  SmallVector<AliasGroup> aliasGroups;
  analyseAllocsForPacking(funcOp, allocs, aliasGroups);
  // If there is 1 or less alias group there is nothing to do.
  if (aliasGroups.size() <= 1)
    return;

  // Pack all the allocations into one i8 alloc.
  // We may need to add extra barriers to make sure we are done writting or
  // reading from the previous alias group before starting a new one.
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    for (Operation *alloc : aliasGroups[i]) {
      addBarrier(funcOp, alloc, aliasGroups[i]);
    }
  }

  OpBuilder builder(funcOp.getContext());
  packAllocs(builder, funcOp, aliasGroups);
}

} // namespace iree_compiler
} // namespace mlir
