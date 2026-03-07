// RUN: iree-opt --pass-pipeline="builtin.module(any(iree-codegen-gpu-map-vector-distribution),canonicalize,cse)" --split-input-file %s | FileCheck %s

// ============================================================================
// 1D contiguous read: 32 threads, each thread loads 4 contiguous f16 elements.
// Layout: (32, 4) : (1, 0)   (lex: thread-outer, value-inner)
//   thread leaf (32), tid_stride=1, data_stride=4;
//   value leaf (4), data_stride=1.
// offset[0] = (tid % 32) * 4 + vid * 1
// Distributed shape: vector<4xf16>  (single contiguous load)
// ============================================================================

#layout_1d_contig = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @read_1d_contiguous
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[REM:.*]] = arith.remui %[[TID]], %[[C32]] : index
// CHECK:       %[[OFF:.*]] = arith.muli %[[REM]], %[[C4]] : index
// CHECK:       %[[READ:.*]] = vector.transfer_read %arg0[%[[OFF]]], %[[CST]] {in_bounds = [true]} : memref<128xf16>, vector<4xf16>
// CHECK:       iree_vector_ext.to_simd %[[READ]] : vector<4xf16> -> vector<128xf16>
func.func @read_1d_contiguous(%arg0: memref<128xf16>) -> vector<128xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_1d_contig) : vector<128xf16>
  func.return %rootl : vector<128xf16>
}

// -----

// ============================================================================
// 1D scalar read: 32 threads, each thread loads 4 non-contiguous f16 elements
// at stride 32.
// Layout: (4, 32) : (0, 1)   (lex: value-outer, thread-inner)
//   value leaf (4), data_stride=32;
//   thread leaf (32), tid_stride=1, data_stride=1.
// offset[0] = tid * 1 + vid * 32
// Distributed shape: vector<4xf16>  (4 scalar loads)
// ============================================================================

#layout_1d_scalar = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// CHECK-LABEL: @read_1d_scalar
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C96:.*]] = arith.constant 96 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[T:.*]] = arith.remui %[[TID]], %[[C32]] : index
//   Scalar load 0: at T
// CHECK:       vector.transfer_read %arg0[%[[T]]], {{.*}} : memref<128xf16>, vector<1xf16>
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [0]
//   Scalar load 1: at T + 32
// CHECK:       %[[OFF1:.*]] = arith.addi %[[T]], %[[C32]]
// CHECK:       vector.transfer_read %arg0[%[[OFF1]]]
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [1]
//   Scalar load 2: at T + 64
// CHECK:       %[[OFF2:.*]] = arith.addi %[[T]], %[[C64]]
// CHECK:       vector.transfer_read %arg0[%[[OFF2]]]
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [2]
//   Scalar load 3: at T + 96
// CHECK:       %[[OFF3:.*]] = arith.addi %[[T]], %[[C96]]
// CHECK:       vector.transfer_read %arg0[%[[OFF3]]]
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [3]
// CHECK:       iree_vector_ext.to_simd {{.*}} : vector<4xf16> -> vector<128xf16>
func.func @read_1d_scalar(%arg0: memref<128xf16>) -> vector<128xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_1d_scalar) : vector<128xf16>
  func.return %rootl : vector<128xf16>
}

// -----

// ============================================================================
// 2D read with contiguous last dim.
// Layout: ((4, 2), (8, 4)) : ((1, 0), (4, 0))   (lex: thread-outer, value-inner)
//   dim 0 (8): thread(4) tid_stride=1, data_stride=2; value(2), data_stride=1
//   dim 1 (32): thread(8) tid_stride=4, data_stride=4; value(4), data_stride=1
// offset[0] = (tid % 4) * 2 + vid0
// offset[1] = ((tid / 4) % 8) * 4 + vid1
// Distributed shape: vector<2x4xf16>  (2 x vector<4xf16> contiguous loads)
// ============================================================================

#layout_2d_contig = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @read_2d_contiguous
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
//   Thread offsets: dim0 = (tid % 4) * 2, dim1 = ((tid / 4) % 8) * 4
// CHECK-DAG:   %[[REM0:.*]] = arith.remui %[[TID]], %[[C4]] : index
// CHECK-DAG:   %[[TOFF0:.*]] = arith.muli %[[REM0]], %[[C2]] : index
// CHECK-DAG:   %[[DIV1:.*]] = arith.divui %[[TID]], %[[C4]] : index
// CHECK-DAG:   %[[REM1:.*]] = arith.remui %[[DIV1]], %[[C8]] : index
// CHECK-DAG:   %[[TOFF1:.*]] = arith.muli %[[REM1]], %[[C4]] : index
//   Iteration 0: read at [TOFF0, TOFF1]
// CHECK:       vector.transfer_read %arg0[%[[TOFF0]], %[[TOFF1]]], {{.*}} : memref<128x128xf16>, vector<4xf16>
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [0, 0]
//   Iteration 1: read at [TOFF0 + 1, TOFF1]
// CHECK:       %[[ROW1:.*]] = arith.addi %[[TOFF0]], %[[C1]]
// CHECK:       vector.transfer_read %arg0[%[[ROW1]], %[[TOFF1]]], {{.*}} : memref<128x128xf16>, vector<4xf16>
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [1, 0]
// CHECK:       iree_vector_ext.to_simd {{.*}} : vector<2x4xf16> -> vector<8x32xf16>
func.func @read_2d_contiguous(%arg0: memref<128x128xf16>) -> vector<8x32xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<8x32xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_2d_contig) : vector<8x32xf16>
  func.return %rootl : vector<8x32xf16>
}

// -----

// ============================================================================
// 2D read with permutation map: reading from a 4D memref with last two dims.
// Same layout as above.
// ============================================================================

#layout_2d_perm = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @read_2d_with_permutation_map
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
//   Two vector loads from the 4D memref with projected dims:
// CHECK:       vector.transfer_read {{.*}} memref<4x4x128x128xf16>, vector<4xf16>
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [0, 0]
// CHECK:       vector.transfer_read {{.*}} memref<4x4x128x128xf16>, vector<4xf16>
// CHECK:       vector.insert_strided_slice {{.*}} {offsets = [1, 0]
// CHECK:       iree_vector_ext.to_simd {{.*}} : vector<2x4xf16> -> vector<8x32xf16>
func.func @read_2d_with_permutation_map(%a: index, %b: index, %arg0: memref<4x4x128x128xf16>) -> vector<8x32xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%a, %b, %c0, %c0], %cst
    {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
    : memref<4x4x128x128xf16>, vector<8x32xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_2d_perm) : vector<8x32xf16>
  func.return %rootl : vector<8x32xf16>
}

// -----

// ============================================================================
// 1D contiguous write: 32 threads, each thread writes 4 contiguous f16 elems.
// Layout: (32, 4) : (1, 0)
// Distributed shape: vector<4xf16>  (single contiguous store)
// ============================================================================

#layout_w1d_contig = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @write_1d_contiguous
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[REM:.*]] = arith.remui %[[TID]], %[[C32]] : index
// CHECK:       %[[OFF:.*]] = arith.muli %[[REM]], %[[C4]] : index
// CHECK:       %[[DIST:.*]] = iree_vector_ext.to_simt %arg1 : vector<128xf16> -> vector<4xf16>
// CHECK:       vector.transfer_write %[[DIST]], %arg0[%[[OFF]]] {in_bounds = [true]} : vector<4xf16>, memref<128xf16>
func.func @write_1d_contiguous(%arg0: memref<128xf16>, %arg1: vector<128xf16>)
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %v = iree_vector_ext.to_layout %arg1 to layout(#layout_w1d_contig) : vector<128xf16>
  vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16>
  func.return
}

// -----

// ============================================================================
// 1D scalar write: 32 threads, each thread writes 4 non-contiguous f16 elems
// at stride 32.
// Layout: (4, 32) : (0, 1)
// Distributed shape: vector<4xf16>  (4 scalar stores)
// ============================================================================

#layout_w1d_scalar = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// CHECK-LABEL: @write_1d_scalar
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C96:.*]] = arith.constant 96 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[T:.*]] = arith.remui %[[TID]], %[[C32]] : index
// CHECK:       %[[DIST:.*]] = iree_vector_ext.to_simt {{.*}} -> vector<4xf16>
//   Scalar store 0: at T
// CHECK:       %[[S0:.*]] = vector.extract_strided_slice %[[DIST]] {offsets = [0], sizes = [1]
// CHECK:       vector.transfer_write %[[S0]], %arg0[%[[T]]] {in_bounds = [true]} : vector<1xf16>
//   Scalar store 1: at T + 32
// CHECK:       %[[OFF1:.*]] = arith.addi %[[T]], %[[C32]]
// CHECK:       %[[S1:.*]] = vector.extract_strided_slice %[[DIST]] {offsets = [1], sizes = [1]
// CHECK:       vector.transfer_write %[[S1]], %arg0[%[[OFF1]]] {in_bounds = [true]} : vector<1xf16>
//   Scalar store 2: at T + 64
// CHECK:       %[[OFF2:.*]] = arith.addi %[[T]], %[[C64]]
// CHECK:       %[[S2:.*]] = vector.extract_strided_slice %[[DIST]] {offsets = [2], sizes = [1]
// CHECK:       vector.transfer_write %[[S2]], %arg0[%[[OFF2]]] {in_bounds = [true]} : vector<1xf16>
//   Scalar store 3: at T + 96
// CHECK:       %[[OFF3:.*]] = arith.addi %[[T]], %[[C96]]
// CHECK:       %[[S3:.*]] = vector.extract_strided_slice %[[DIST]] {offsets = [3], sizes = [1]
// CHECK:       vector.transfer_write %[[S3]], %arg0[%[[OFF3]]] {in_bounds = [true]} : vector<1xf16>
func.func @write_1d_scalar(%arg0: memref<128xf16>, %arg1: vector<128xf16>)
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %v = iree_vector_ext.to_layout %arg1 to layout(#layout_w1d_scalar) : vector<128xf16>
  vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16>
  func.return
}

// -----

// ============================================================================
// 2D contiguous write with contiguous last dim.
// Layout: ((4, 2), (8, 4)) : ((1, 0), (4, 0))
// Distributed shape: vector<2x4xf16>  (2 x vector<4xf16> contiguous stores)
// ============================================================================

#layout_w2d_contig = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @write_2d_contiguous
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK-DAG:   %[[REM0:.*]] = arith.remui %[[TID]], %[[C4]] : index
// CHECK-DAG:   %[[TOFF0:.*]] = arith.muli %[[REM0]], %[[C2]] : index
// CHECK-DAG:   %[[DIV1:.*]] = arith.divui %[[TID]], %[[C4]] : index
// CHECK-DAG:   %[[REM1:.*]] = arith.remui %[[DIV1]], %[[C8]] : index
// CHECK-DAG:   %[[TOFF1:.*]] = arith.muli %[[REM1]], %[[C4]] : index
// CHECK:       %[[DIST:.*]] = iree_vector_ext.to_simt {{.*}} -> vector<2x4xf16>
//   Iteration 0: write at [threadOff0, threadOff1]
// CHECK:       vector.extract %[[DIST]][0]
// CHECK:       vector.transfer_write {{.*}}, %arg0[%[[TOFF0]], %[[TOFF1]]] {in_bounds = [true]} : vector<4xf16>, memref<128x128xf16>
//   Iteration 1: write at [threadOff0 + 1, threadOff1]
// CHECK:       vector.extract %[[DIST]][1]
// CHECK:       vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf16>, memref<128x128xf16>
func.func @write_2d_contiguous(%arg0: memref<128x128xf16>, %arg1: vector<8x32xf16>)
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %v = iree_vector_ext.to_layout %arg1 to layout(#layout_w2d_contig) : vector<8x32xf16>
  vector.transfer_write %v, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf16>, memref<128x128xf16>
  func.return
}

// -----

// ============================================================================
// 2D write with permutation map: writing to a 4D memref with last two dims.
// Same layout as above.
// ============================================================================

#layout_w2d_perm = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @write_2d_with_permutation_map
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
//   Two vector stores to the 4D memref with projected dims:
// CHECK:       vector.transfer_write {{.*}} memref<4x4x128x128xf16>
// CHECK:       vector.transfer_write {{.*}} memref<4x4x128x128xf16>
func.func @write_2d_with_permutation_map(%a: index, %b: index, %arg0: memref<4x4x128x128xf16>, %arg1: vector<8x32xf16>)
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %v = iree_vector_ext.to_layout %arg1 to layout(#layout_w2d_perm) : vector<8x32xf16>
  vector.transfer_write %v, %arg0[%a, %b, %c0, %c0]
    {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
    : vector<8x32xf16>, memref<4x4x128x128xf16>
  func.return
}

// -----

// ============================================================================
// Complex round-trip: transfer_read → transfer_write with hierarchical modes.
//
// 64×32 tile, 32 threads. Interleaved thread/value modes:
//
//   Dim 0 (64 elems):
//     (2, 2, 2, 2, 2, 2) : (1, 0, 2, 0, 4, 0)
//     Thread leaves (lex): (2):(1), (2):(2), (2):(4) — 8 threads
//       data strides: 32, 8, 2
//     Value leaves  (lex): (2):(0), (2):(0), (2):(0) — 8 values
//       data strides: 16, 4, 1
//
//   Dim 1 (32 elems):
//     (4, 8) : (8, 0)
//     Thread leaf: (4):(8), data_stride=8 — 4 groups at cols 0,8,16,24
//     Value leaf:  (8):(0), data_stride=1 — contiguous
//
// Distributed shape: [2, 2, 2, 8]  (3 value leaves from dim0 + 1 from dim1)
// Last leaf stride = 1 → contiguous vector<8xf16> access.
// 8 iterations of vector<8xf16> loads + 8 stores.
// ============================================================================

#complex = #iree_map.pack_layout<
  ((2, 2, 2, 2, 2, 2), (4, 8)) : ((1, 0, 2, 0, 4, 0), (8, 0))
>

//      CHECK-LABEL: @complex_read_write
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C32:.*]] = arith.constant 32 : index
//
//      CHECK:       %[[TID:.*]] = gpu.thread_id  x
//
//            Dim 0 thread offset (lex): (tid%2)*32 + ((tid/2)%2)*8 + ((tid/4)%2)*2
//  CHECK-DAG:       %[[T0_0:.*]] = arith.remui %[[TID]], %[[C2]]
//  CHECK-DAG:       %[[M0_0:.*]] = arith.muli %[[T0_0]], %[[C32]]
//  CHECK-DAG:       %[[D0_1:.*]] = arith.divui %[[TID]], %[[C2]]
//  CHECK-DAG:       %[[T0_1:.*]] = arith.remui %[[D0_1]], %[[C2]]
//  CHECK-DAG:       %[[M0_1:.*]] = arith.muli %[[T0_1]], %[[C8]]
//  CHECK-DAG:       %[[S0_01:.*]] = arith.addi %[[M0_0]], %[[M0_1]]
//  CHECK-DAG:       %[[D0_2:.*]] = arith.divui %[[TID]], %[[C4]]
//  CHECK-DAG:       %[[T0_2:.*]] = arith.remui %[[D0_2]], %[[C2]]
//  CHECK-DAG:       %[[M0_2:.*]] = arith.muli %[[T0_2]], %[[C2]]
//  CHECK-DAG:       %[[TOFF0:.*]] = arith.addi %[[S0_01]], %[[M0_2]]
//
//            Dim 1 thread offset: ((tid/8)%4)*8
//      CHECK:       %[[D1:.*]] = arith.divui %[[TID]], %[[C8]]
//      CHECK:       %[[R1:.*]] = arith.remui %[[D1]], %[[C4]]
//      CHECK:       %[[TOFF1:.*]] = arith.muli %[[R1]], %[[C8]]
//
//            8 contiguous vector<8xf16> reads:
//      CHECK:       vector.transfer_read %arg0[%[[TOFF0]], %[[TOFF1]]]{{.*}} vector<8xf16>
// CHECK-COUNT-7:    vector.transfer_read %arg0{{.*}} vector<8xf16>
//
//            8 contiguous vector<8xf16> writes:
// CHECK-COUNT-8:    vector.transfer_write {{.*}} vector<8xf16>
func.func @complex_read_write(%src: memref<64x32xf16>, %dst: memref<64x32xf16>)
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %src[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<64x32xf16>, vector<64x32xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#complex) : vector<64x32xf16>
  vector.transfer_write %vl, %dst[%c0, %c0] {in_bounds = [true, true]}
    : vector<64x32xf16>, memref<64x32xf16>
  func.return
}

// -----

// ============================================================================
// Layout conversion via shared memory: to_layout between two different
// layouts on the same 1D vector.
//
// Layout A: 32 threads, each owns 4 contiguous elements.
//   (32, 4) : (1, 0) — thread outer, value inner (lex)
//   thread t owns elements [4t, 4t+3]
//
// Layout B: 32 threads, each owns 4 elements at stride 32.
//   (4, 32) : (0, 1) — value outer, thread inner (lex)
//   thread t owns elements {t, t+32, t+64, t+96}
//
// Conversion requires shared memory: write with A's indexing, read with B's.
// ============================================================================

#layout_a = #iree_map.pack_layout<((32, 4)) : ((1, 0))>
#layout_b = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// CHECK-LABEL: @to_layout_shared_memory_1d
//
//   Shared memory allocation:
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<128xf16, #gpu.address_space<workgroup>>
//
//   WAR barrier:
// CHECK:       gpu.barrier
//
//   Distributed write (layout A: contiguous vector<4xf16>):
// CHECK:       vector.transfer_write {{.*}}, %[[ALLOC]]{{.*}} : vector<4xf16>, memref<128xf16, #gpu.address_space<workgroup>>
//
//   RAW barrier:
// CHECK:       gpu.barrier
//
//   Distributed reads (layout B: 4 scalar reads at stride 32):
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<128xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<128xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<128xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<128xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK:       iree_vector_ext.to_simd {{.*}} : vector<4xf16> -> vector<128xf16>
func.func @to_layout_shared_memory_1d(%arg0: memref<128xf16>) -> vector<128xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %a = iree_vector_ext.to_layout %v to layout(#layout_a) : vector<128xf16>
  %b = iree_vector_ext.to_layout %a to layout(#layout_b) : vector<128xf16>
  func.return %b : vector<128xf16>
}
