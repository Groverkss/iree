# Remaining CuTe Distribution Patterns

Patterns needed for parity with `GPUNestedLayoutDistributionPatterns.cpp`.

## Already Implemented

| Pattern | Op | File |
|---------|-----|------|
| TransferRead | `vector::TransferReadOp` | CuteDistributeMemoryOps.cpp |
| TransferWrite | `vector::TransferWriteOp` | CuteDistributeMemoryOps.cpp |
| ToLayout | `IREE::VectorExt::ToLayoutOp` | CuteDistributeMemoryOps.cpp |
| MultiReduction | `vector::MultiDimReductionOp` | CuteDistributeReductionOps.cpp |
| ShapeCast | `vector::ShapeCastOp` | CuteDistributeGenericOps.cpp |
| Broadcast | `vector::BroadcastOp` | CuteDistributeGenericOps.cpp |
| Transpose | `vector::TransposeOp` | CuteDistributeGenericOps.cpp |
| Step | `vector::StepOp` | CuteDistributeGenericOps.cpp |
| InnerTiled | `IREE::Codegen::InnerTiledOp` | CuteDistributeInnerTiledOps.cpp |

## Skipped (decompose before distribution instead)

| Pattern | Op | Reason |
|---------|-----|--------|
| CreateMask | `vector::CreateMaskOp` | Decompose to step + compare in earlier pass |
| ConstantMask | `vector::ConstantMaskOp` | Decompose to step + compare in earlier pass |

## Missing

| # | Pattern | Op | Description |
|---|---------|-----|-------------|
| 1 | Contract | `vector::ContractionOp` | Distribute matmul-like contractions across threads |
| 2 | TransferGather | `IREE::VectorExt::TransferGatherOp` | Distribute indirect/gather memory reads |
| 3 | MapStore | `IREE::LinalgExt::MapStoreOp` | Distribute element-wise transform + store ops |
