// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-prefetching"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {

/// Returns true if the given `memrefType` has the default numeric address space
/// 0 or a HAL descriptor type address space.
static bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace)
    return true;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(addrSpace);
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0)
    return true;
  return llvm::isa<IREE::HAL::DescriptorTypeAttr>(addrSpace);
}

class LoopPrefetcher {
public:
  bool initializeLoopInfo(scf::ForOp op) {
    forOp = op;
    auto upperBoundCst =
        forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lowerBoundCst =
        forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!upperBoundCst || !lowerBoundCst || !stepCst) {
      return false;
    }
    ub = upperBoundCst.value();
    lb = lowerBoundCst.value();
    step = stepCst.value();

    int64_t numIteration = ceilDiv(ub - lb, step);

    if (numIteration <= 2) {
      return false;
    }

    return true;
  }

  enum class Stage { Read, Compute, Write, None };

  void getReachable(Operation *op, DenseSet<Operation *> &reachable) {
    if (!op)
      return;

    if (reachable.contains(op)) {
      return;
    }

    if (!forOp->isProperAncestor(op)) {
      return;
    }

    op->walk([&](Operation *nested) {
      reachable.insert(op);
      for (Value val : nested->getOperands()) {
        getReachable(val.getDefiningOp(), reachable);
      }
    });
  }

  void initializeStages() {
    DenseSet<Operation *> readReachable;
    DenseSet<Operation *> writeReachable;
    DenseSet<Operation *> computeReachable;

    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
        getReachable(read, readReachable);
      }
      if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
        getReachable(write, writeReachable);
      }
      if (auto compute = dyn_cast<scf::YieldOp>(op)) {
        getReachable(compute, computeReachable);
      }
    }

    for (auto &op : forOp.getBody()->getOperations()) {
      if (readReachable.contains(&op)) {
        readStage.push_back(&op);
      }
      if (writeReachable.contains(&op)) {
        writeStage.push_back(&op);
      }
      if (computeReachable.contains(&op)) {
        computeStage.push_back(&op);
      }
    }
  }

  /// Clone `op` and call `callback` on the cloned op's oeprands as well as any
  /// operands of nested ops that:
  /// 1) aren't defined within the new op or
  /// 2) are block arguments.
  static Operation *
  cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                         function_ref<void(OpOperand *newOperand)> callback) {
    Operation *clone = rewriter.clone(*op);
    for (OpOperand &operand : clone->getOpOperands())
      callback(&operand);
    return clone;
  }

  void emitPrologue(RewriterBase &rewriter) {
    // Read(0).
    // Write(0).

    // Get a value mapping for iv = lb.
    Value iv = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), lb);
    mapping.map(forOp.getInductionVar(), iv);

    // Read(0).
    for (Operation *op : readStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      // Map read operations to new read operations.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    // Write(0).
    for (Operation *op : writeStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      // If a mapping for any results already exists, move on, otherwise,
      // add a new mapping.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (!mapping.contains(op->getResult(i))) {
          mapping.map(op->getResult(i), newOp->getResult(i));
        }
      }
    }

    mapping.clear();
  }

  scf::ForOp createKernelLoop(RewriterBase &rewriter) {
    auto newUb =
        rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), ub - step);
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), newUb, forOp.getStep(),
        forOp.getInitArgs());

    // When there are no iter args, the loop body terminator will be created.
    // Since we always create it below, remove the terminator if it was created.
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());
    return newForOp;
  }

  void createKernel(scf::ForOp newForOp, RewriterBase &rewriter) {
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    IRMapping readWriteMap;
    IRMapping computeMap;

    Value iv = newForOp.getInductionVar();
    Value leap = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), step);
    Value ivPlusOne = rewriter.create<arith::AddIOp>(forOp.getLoc(), iv, leap);

    computeMap.map(forOp.getInductionVar(), iv);
    readWriteMap.map(forOp.getInductionVar(), ivPlusOne);

    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      computeMap.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
      readWriteMap.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    }

    // Read(i+1).
    for (Operation *op : readStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (readWriteMap.contains(newOperand->get())) {
              newOperand->set(readWriteMap.lookup(newOperand->get()));
            }
          });

      // Map read operations to new read operations.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        readWriteMap.map(op->getResult(i), newOp->getResult(i));
      }
    }

    rewriter.create<gpu::BarrierOp>(forOp.getLoc());

    // Compute(i).
    for (Operation *op : computeStage) {
      // Ignore the yield for now.
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        continue;

      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (computeMap.contains(newOperand->get())) {
              newOperand->set(computeMap.lookup(newOperand->get()));
            }
          });

      // Map read operations to new read operations.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        computeMap.map(op->getResult(i), newOp->getResult(i));
      }
    }

    // barrier
    rewriter.create<gpu::BarrierOp>(forOp.getLoc());

    // Write(i+1).
    for (Operation *op : writeStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (readWriteMap.contains(newOperand->get())) {
              newOperand->set(readWriteMap.lookup(newOperand->get()));
            }
          });

      // If a mapping for any results already exists, move on, otherwise,
      // add a new mapping.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (!readWriteMap.contains(op->getResult(i))) {
          readWriteMap.map(op->getResult(i), newOp->getResult(i));
        }
      }
    }

    // yield
    for (auto op : computeStage) {
      if (auto yeild = dyn_cast<scf::YieldOp>(op)) {
        cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
          if (computeMap.contains(newOperand->get())) {
            newOperand->set(computeMap.lookup(newOperand->get()));
          }
        });
      }
    }
  }

  SmallVector<Value> emitEpilogue(scf::ForOp newForOp, RewriterBase &rewriter) {
    rewriter.create<gpu::BarrierOp>(forOp.getLoc());

    // Compute(lb + (numIt - 1) * step).
    Value lastIter = rewriter.create<arith::ConstantIndexOp>(
        forOp.getLoc(), lb + step * ((((ub - 1) - lb) / step)));
    mapping.map(forOp.getInductionVar(), lastIter);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0; i < newForOp.getNumResults(); ++i) {
      mapping.map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    // Compute(lastIter).
    for (Operation *op : computeStage) {
      // Ignore the yield for now.
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        continue;
      }

      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      // Map read operations to new read operations.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    // Get the result as mapping of yieldop.
    SmallVector<Value> newRes;
    for (Operation *op : computeStage) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        for (auto res : yieldOp.getResults()) {
          newRes.push_back(mapping.lookup(res));
        }
        break;
      }
    }

    return newRes;
  }

private:
  IRMapping mapping;
  scf::ForOp forOp;
  int64_t ub, lb, step;

  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;
};

namespace {

FailureOr<scf::ForOp> applyPrefetching(scf::ForOp forOp) {
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);

  LoopPrefetcher prefetcher;
  if (!prefetcher.initializeLoopInfo(forOp)) {
    return failure();
  }
  prefetcher.initializeStages();

  prefetcher.emitPrologue(rewriter);

  scf::ForOp newForOp = prefetcher.createKernelLoop(rewriter);
  prefetcher.createKernel(newForOp, rewriter);

  rewriter.setInsertionPointAfter(newForOp);
  auto results = prefetcher.emitEpilogue(newForOp, rewriter);

  rewriter.replaceOp(forOp, results);
  return newForOp;
}

struct GPUPrefetchingPass : public GPUPrefetchingBase<GPUPrefetchingPass> {
  GPUPrefetchingPass() {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      (void)applyPrefetching(forOp);
    }
  }
};
} // namespace

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp) {
  return applyPrefetching(forOp);
}

/// Pass options
/// epiloguePeeling - try enable/disable epilogue peeling.
/// true  : Peel epilogue (no additional checks required)
/// false : Try and use unpeeled epilogue (check if predication is supported
/// is avialable)
std::unique_ptr<OperationPass<func::FuncOp>> createGPUPrefetchingPass() {
  return std::make_unique<GPUPrefetchingPass>();
}

} // namespace iree_compiler
} // namespace mlir
