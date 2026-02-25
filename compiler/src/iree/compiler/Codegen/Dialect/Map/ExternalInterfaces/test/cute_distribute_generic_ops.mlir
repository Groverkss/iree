// RUN: iree-opt --pass-pipeline="builtin.module(any(iree-codegen-gpu-cute-vector-distribution),canonicalize,cse)" --split-input-file %s | FileCheck %s

// ============================================================================
// Shape cast: expand 1D → 2D.
//
// Source: vector<128xf16>, layout (32, 4) : (1, 0)
//   distributed: vector<4xf16>
//
// After reshape to (4, 32):
//   reshapeLayout expands via compose producing a different layout.
//   The reshaped layout is incompatible with the source, so shared memory
//   is used for redistribution.
// ============================================================================

#layout_sc_1d = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @shape_cast_expand_1d_to_2d
//   Read produces distributed vector<4xf16> with source layout:
// CHECK:       vector.transfer_read {{.*}} : memref<128xf16>, vector<4xf16>
// CHECK:       vector.shape_cast {{.*}} : vector<4xf16> to vector<1x4xf16>
func.func @shape_cast_expand_1d_to_2d(%arg0: memref<128xf16>) -> vector<4x32xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_sc_1d) : vector<128xf16>
  %sc = vector.shape_cast %vl : vector<128xf16> to vector<4x32xf16>
  func.return %sc : vector<4x32xf16>
}

// -----

// ============================================================================
// Shape cast: contract 2D → 1D.
//
// Source: vector<4x32xf16>, layout ((4), (8, 4)) : ((8), (1, 0))
//   dim 0: 4 threads at tid_stride=8, no values → distributed dim size 1
//   dim 1: 8 threads at tid_stride=1, data_stride=4; value(4) data_stride=1
//   distributed: vector<1x4xf16>
//
// After reshape to (128):
//   The contracted layout is incompatible with the source, so shared memory
//   is used for redistribution.
// ============================================================================

#layout_sc_2d = #iree_map.pack_layout<((4), (8, 4)) : ((8), (1, 0))>

// CHECK-LABEL: @shape_cast_contract_2d_to_1d
//   Read produces distributed vector<1x4xf16>:
// CHECK:       vector.transfer_read {{.*}} : memref<128x128xf16>, vector<4xf16>
// CHECK:       vector.shape_cast {{.*}} : vector<1x4xf16> to vector<4xf16>
func.func @shape_cast_contract_2d_to_1d(%arg0: memref<128x128xf16>) -> vector<128xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<128x128xf16>, vector<4x32xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_sc_2d) : vector<4x32xf16>
  %sc = vector.shape_cast %vl : vector<4x32xf16> to vector<128xf16>
  func.return %sc : vector<128xf16>
}

// -----

// ============================================================================
// Broadcast: scalar → 1D vector.
// Layout: (32, 4) : (1, 0)
// Distributed shape: vector<4xf16>
// ============================================================================

#layout_bcast_scalar = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @broadcast_scalar
// CHECK:       %[[BCAST:.*]] = vector.broadcast %arg0 : f16 to vector<4xf16>
// CHECK:       iree_vector_ext.to_simd %[[BCAST]] : vector<4xf16> -> vector<128xf16>
func.func @broadcast_scalar(%arg0: f16) -> vector<128xf16>
    attributes { workgroup_size = [32 : index] } {
  %b = vector.broadcast %arg0 : f16 to vector<128xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_bcast_scalar) : vector<128xf16>
  func.return %bl : vector<128xf16>
}

// -----

// ============================================================================
// Broadcast: 1D → 2D (add leading dim).
// Dest layout: ((4, 2), (8, 4)) : ((1, 0), (4, 0))
//   Dest distributed shape: vector<2x4xf16>
// Source layout (backward-propagated via project): (8, 4) : (4, 0)
//   Source distributed shape: vector<4xf16>
// Distributed broadcast: vector<4xf16> → vector<2x4xf16>
// ============================================================================

#layout_bcast_2d = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @broadcast_vector_add_dim
// CHECK:       %[[READ:.*]] = vector.transfer_read {{.*}} : memref<32xf16>, vector<4xf16>
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[READ]] : vector<4xf16> to vector<2x4xf16>
// CHECK:       iree_vector_ext.to_simd %[[BCAST]] : vector<2x4xf16> -> vector<8x32xf16>
func.func @broadcast_vector_add_dim(%arg0: memref<32xf16>) -> vector<8x32xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<32xf16>, vector<32xf16>
  %b = vector.broadcast %v : vector<32xf16> to vector<8x32xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_bcast_2d) : vector<8x32xf16>
  func.return %bl : vector<8x32xf16>
}

// -----

// ============================================================================
// Transpose: simple 2D swap.
// Layout: ((4, 2), (8, 4)) : ((1, 0), (4, 0))
//   Distributed shape: vector<2x4xf16>
//   Each mode has 1 value leaf → expanded perm = [1, 0]
// ============================================================================

#layout_tr_2d = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @transpose_2d
// CHECK:       vector.transfer_read {{.*}} : memref<8x32xf16>, vector<4xf16>
// CHECK:       vector.transpose %{{.*}}, [1, 0] : vector<2x4xf16> to vector<4x2xf16>
func.func @transpose_2d(%arg0: memref<8x32xf16>) -> vector<32x8xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<8x32xf16>, vector<8x32xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_tr_2d) : vector<8x32xf16>
  %t = vector.transpose %vl, [1, 0] : vector<8x32xf16> to vector<32x8xf16>
  func.return %t : vector<32x8xf16>
}

// -----

// ============================================================================
// Transpose with multi-leaf mode.
// Layout: ((2,2,2,2,2,2), (4,8)) : ((1,0,2,0,4,0), (8,0))
//   Dim 0: 3 value leaves → distributed dims [0, 1, 2]
//   Dim 1: 1 value leaf  → distributed dim [3]
//   Distributed shape: vector<2x2x2x8xf16>
// Perm [1, 0] → expanded [3, 0, 1, 2]
// ============================================================================

#layout_tr_complex = #iree_map.pack_layout<
  ((2, 2, 2, 2, 2, 2), (4, 8)) : ((1, 0, 2, 0, 4, 0), (8, 0))
>

// CHECK-LABEL: @transpose_multi_leaf
// CHECK:       vector.transpose %{{.*}}, [3, 0, 1, 2] : vector<2x2x2x8xf16> to vector<8x2x2x2xf16>
func.func @transpose_multi_leaf(%arg0: memref<64x32xf16>) -> vector<32x64xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<64x32xf16>, vector<64x32xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_tr_complex) : vector<64x32xf16>
  %t = vector.transpose %vl, [1, 0] : vector<64x32xf16> to vector<32x64xf16>
  func.return %t : vector<32x64xf16>
}

// -----

// ============================================================================
// Step: contiguous value leaves.
// Layout: (32, 4) : (1, 0)
//   thread leaf (32), tid_stride=1, data_stride=4
//   value leaf (4), data_stride=1
// Distributed shape: vector<4xindex>
// Value offsets: [0, 1, 2, 3]
// Thread offset: (tid % 32) * 4
// ============================================================================

#layout_step_contig = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @step_contiguous
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[REM:.*]] = arith.remui %[[TID]], %[[C32]] : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[REM]], %[[C4]] : index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[MUL]] : index to vector<4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4xindex> -> vector<128xindex>
func.func @step_contiguous() -> vector<128xindex>
    attributes { workgroup_size = [32 : index] } {
  %step = vector.step : vector<128xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_contig) : vector<128xindex>
  func.return %sl : vector<128xindex>
}

// -----

// ============================================================================
// Step: strided value leaves.
// Layout: (4, 8) : (0, 1)
//   value leaf (4), data_stride=8
//   thread leaf (8), tid_stride=1, data_stride=1
// Distributed shape: vector<4xindex>
// Value offsets: [0, 8, 16, 24]
// Thread offset: tid % 8
// ============================================================================

#layout_step_strided = #iree_map.pack_layout<((4, 8)) : ((0, 1))>

// CHECK-LABEL: @step_strided
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[REM:.*]] = arith.remui %[[TID]], %[[C8]] : index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[REM]] : index to vector<4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4xindex> -> vector<32xindex>
func.func @step_strided() -> vector<32xindex>
    attributes { workgroup_size = [8 : index] } {
  %step = vector.step : vector<32xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_strided) : vector<32xindex>
  func.return %sl : vector<32xindex>
}

// -----

// ============================================================================
// Step: multiple value leaves (2D distributed shape).
// Layout: (4, 2, 4) : (0, 1, 0)
//   value leaf (4), data_stride=8
//   thread leaf (2), tid_stride=1, data_stride=4
//   value leaf (4), data_stride=1
// Distributed shape: vector<4x4xindex>
// Value offsets: val[i][j] = i*8 + j
// Thread offset: (tid % 2) * 4
// ============================================================================

#layout_step_multi = #iree_map.pack_layout<((4, 2, 4)) : ((0, 1, 0))>

// CHECK-LABEL: @step_multi_value_leaf
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<{{\[\[}}0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24, 25, 26, 27]]> : vector<4x4xindex>
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[REM:.*]] = arith.remui %[[TID]], %[[C2]] : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[REM]], %[[C4]] : index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[MUL]] : index to vector<4x4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4x4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4x4xindex> -> vector<32xindex>
func.func @step_multi_value_leaf() -> vector<32xindex>
    attributes { workgroup_size = [2 : index] } {
  %step = vector.step : vector<32xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_multi) : vector<32xindex>
  func.return %sl : vector<32xindex>
}
