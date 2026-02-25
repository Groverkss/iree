// RUN: iree-opt --pass-pipeline="builtin.module(any(iree-codegen-gpu-cute-vector-distribution),canonicalize,cse)" --split-input-file %s | FileCheck %s

// ============================================================================
// 1D full reduction: 32 threads, each owns 4 contiguous f16 elements.
// Layout: (32, 4) : (1, 0)
// Reduction: vector<128xf16> → f16 (add)
//
// Step 1 (local): multi_reduction on distributed vector<4xf16> → f16 scalar
// Step 2 (shuffle): gpu.subgroup_reduce with cluster_size=32, cluster_stride=1
// Step 3: combine with accumulator
// ============================================================================

#layout_reduce_1d = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @reduce_1d_full
//   Local reduce:
// CHECK:       %[[LOCAL:.*]] = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [0] : vector<4xf16> to f16
//   Thread shuffle:
// CHECK:       %[[SHUFFLE:.*]] = gpu.subgroup_reduce add %[[LOCAL]] cluster(size = 32) : (f16) -> f16
//   Combine with accumulator:
// CHECK:       arith.addf %[[SHUFFLE]], %{{.*}} : f16
func.func @reduce_1d_full(%arg0: memref<128xf16>, %acc: f16) -> f16
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_reduce_1d) : vector<128xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [0] : vector<128xf16> to f16
  func.return %r : f16
}

// -----

// ============================================================================
// 2D partial reduction: reduce dim 1 only, keep dim 0.
// Layout: ((4, 2), (8, 4)) : ((1, 0), (4, 0))
//   dim 0: thread(4) × value(2) = 8     (parallel)
//   dim 1: thread(8) × value(4) = 32    (reduction)
// Total: 32 threads.
//
// Distributed shape: vector<2x4xf16>
// After local reduce of dim1 value leaves: vector<2xf16>
// Thread shuffle for dim 1: cluster_size=8, cluster_stride=4
// ============================================================================

#layout_reduce_2d = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @reduce_2d_partial
//   Local reduce (reduces dim 1 = distributed dim 1):
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<2x4xf16> to vector<2xf16>
//   Thread shuffle (2 elements, each shuffled with cluster_size=8, stride=4):
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 4) : (f16) -> f16
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 4) : (f16) -> f16
//   Combine with accumulator:
// CHECK:       arith.addf %{{.*}}, %{{.*}} : vector<2xf16>
func.func @reduce_2d_partial(%arg0: memref<8x32xf16>, %acc: vector<8xf16>) -> vector<8xf16>
    attributes { workgroup_size = [32 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf16>, vector<8x32xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_reduce_2d) : vector<8x32xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [1] : vector<8x32xf16> to vector<8xf16>
  func.return %r : vector<8xf16>
}

// -----

// ============================================================================
// 3D multi-dim reduction with two reduction dims and sequential shuffles.
//
// vector<4x4x8xf16>, reduce dims 1 and 2, keep dim 0.
// Layout: ((2, 2), (2, 2), (4, 2)) : ((1, 0), (2, 0), (4, 0))
//   dim 0 (4): thread(2) tid_stride=1 + value(2) = 4   (parallel)
//   dim 1 (4): thread(2) tid_stride=2 + value(2) = 4   (reduced)
//   dim 2 (8): thread(4) tid_stride=4 + value(2) = 8   (reduced)
//   Total: 2*2*4 = 16 threads.
//
// Distributed source: vector<2x2x2xf16>
// After local reduce [1, 2]: vector<2xf16>
//
// Shuffle: dims 1,2 → cluster(8,2)
// ============================================================================

#layout_reduce_3d = #iree_map.pack_layout<
  ((2, 2), (2, 2), (4, 2)) : ((1, 0), (2, 0), (4, 0))
>

// CHECK-LABEL: @reduce_3d_multi_dim
//   Local reduce (dims 1,2 → distributed dims 1,2):
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1, 2] : vector<2x2x2xf16> to vector<2xf16>
//   Thread shuffles: dims 1 and 2 coalesce → (8:2):
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 2) : (f16) -> f16
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 2) : (f16) -> f16
//   Combine with accumulator:
// CHECK:       arith.addf %{{.*}}, %{{.*}} : vector<2xf16>
func.func @reduce_3d_multi_dim(%arg0: memref<4x4x8xf16>, %acc: vector<4xf16>) -> vector<4xf16>
    attributes { workgroup_size = [16 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]}
    : memref<4x4x8xf16>, vector<4x4x8xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_reduce_3d) : vector<4x4x8xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [1, 2] : vector<4x4x8xf16> to vector<4xf16>
  func.return %r : vector<4xf16>
}

// -----

// ============================================================================
// 1D full reduction across subgroups.
// Layout: (64, 2) : (1, 0)
//   thread leaf (64) tid_stride=1, data_stride=2;
//   value leaf (2) data_stride=1 — contiguous pair per thread.
//   sg=32 → 2 subgroups.
//
// Phase 1: Local reduce 2 values → scalar.
//   Shuffle within subgroup: cluster_size=32, stride=1.
// Phase 2: Alloc shared memory [2].
//   Write: across-subgroup threads map to buffer position.
// Phase 3: Read all 2 partial results per thread.
//   Second local reduce of 2 values → scalar. Combine with acc.
// ============================================================================

#layout_cross_1d = #iree_map.pack_layout<((64, 2)) : ((1, 0))>

// CHECK-LABEL: @reduce_1d_cross_subgroup
//   Local reduce:
// CHECK:       %[[LOCAL:.*]] = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [0] : vector<2xf16> to f16
//   Within-subgroup shuffle:
// CHECK:       %[[SHUF:.*]] = gpu.subgroup_reduce add %[[LOCAL]] cluster(size = 32) : (f16) -> f16
//   Shared memory buffer:
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<2xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Write partial result:
// CHECK:       vector.transfer_write {{.*}}, %[[ALLOC]]{{.*}} : vector<1xf16>, memref<2xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Read all partial results:
// CHECK:       %[[READ:.*]] = vector.transfer_read %[[ALLOC]]{{.*}} : memref<2xf16, #gpu.address_space<workgroup>>, vector<2xf16>
//   Second reduction (local only, no shuffle):
// CHECK:       %[[RED2:.*]] = vector.multi_reduction <add>, %[[READ]], %{{.*}} [0] : vector<2xf16> to f16
//   Combine with accumulator:
// CHECK:       arith.addf %[[RED2]], %{{.*}} : f16
func.func @reduce_1d_cross_subgroup(%arg0: memref<128xf16>, %acc: f16) -> f16
    attributes { workgroup_size = [64 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_cross_1d) : vector<128xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [0] : vector<128xf16> to f16
  func.return %r : f16
}

// -----

// ============================================================================
// 2D partial reduction across subgroups: reduce dim 1 only, keep dim 0.
// Layout: ((4, 2), (32, 2)) : ((1, 0), (4, 0))
//   dim 0 (8): thread(4) tid_stride=1 + value(2)   (parallel)
//   dim 1 (64): thread(32) tid_stride=4 + value(2)  (reduction)
//   128 threads total, sg=32.
//
// Split: dim 0: within=4, across=1. dim 1: within=8, across=4.
//
// Phase 1: Local reduce dim 1 value leaves: vector<2x2xf16> → vector<2xf16>.
//   Shuffle dim 1: cluster_size=8, cluster_stride=4.
// Phase 2: Buffer shape [8, 4].
// Phase 3: Read: second local reduce of 4 partial results.
// ============================================================================

#layout_cross_2d = #iree_map.pack_layout<((4, 2), (32, 2)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @reduce_2d_cross_subgroup
//   Local reduce dim 1:
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<2x2xf16> to vector<2xf16>
//   Within-subgroup shuffle (2 elements, cluster_size=8, stride=4):
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 4) : (f16) -> f16
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 8, stride = 4) : (f16) -> f16
//   Shared memory buffer [8, 4]:
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<8x4xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Write partial results (2 writes for 2 non-reduced dim value leaves):
// CHECK:       vector.transfer_write {{.*}}, %[[ALLOC]]{{.*}} : vector<1xf16>, memref<8x4xf16, #gpu.address_space<workgroup>>
// CHECK:       vector.transfer_write {{.*}}, %[[ALLOC]]{{.*}} : vector<1xf16>, memref<8x4xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Read all partial results (2 reads of vector<4xf16>):
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<8x4xf16, #gpu.address_space<workgroup>>, vector<4xf16>
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<8x4xf16, #gpu.address_space<workgroup>>, vector<4xf16>
//   Second reduction (local reduce of 4 partial results per non-reduced leaf):
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<2x4xf16> to vector<2xf16>
//   Combine with accumulator:
// CHECK:       arith.addf %{{.*}}, %{{.*}} : vector<2xf16>
func.func @reduce_2d_cross_subgroup(%arg0: memref<8x64xf16>, %acc: vector<8xf16>) -> vector<8xf16>
    attributes { workgroup_size = [128 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<8x64xf16>, vector<8x64xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_cross_2d) : vector<8x64xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [1] : vector<8x64xf16> to vector<8xf16>
  func.return %r : vector<8xf16>
}

// -----

// ============================================================================
// 4D multi-dim reduction across subgroups with multi-mode thread grid.
// Layout: ((2,2,2), (4,2), (2,2), (4,4)) : ((0,1,0), (2,128), (8,64), (16,0))
//   dim 0 (8): thread(2) tid_stride=1 + value(2,2)          (parallel)
//   dim 1 (8): thread(4) tid_stride=2, thread(2) tid_stride=128 (reduction)
//   dim 2 (4): thread(2) tid_stride=8, thread(2) tid_stride=64  (parallel)
//   dim 3 (16): thread(4) tid_stride=16 + value(4)            (reduction)
//   256 threads total, sg=32.
//
// Reduction thread grid (dims 1,3):
//   Thread leaves: (4):(2), (2):(128), (4):(16).
//   Sort desc: (2,4,4):(128,16,2). Coalesce: no merges (4*2≠16, 4*16≠128).
//   Grid = (2,4,4):(128,16,2), size=32.
//
// Subgroup split: K = computeSubgroupTilerSize((2,4,4):(128,16,2), 32).
//   Mode (4):(2): fit=4, maxTid=6. Mode (4):(16): fit=min(4,(25)/16+1)=2,
//   maxTid=22. Mode (2):(128): fit=min(2,(9)/128+1)=1. K=8.
//   logicalDivide(grid, (8):(1)) → within-SG + across-SG(size=4).
//
// Phase 1: Local reduce dims 1,3 value leaves.
// Phase 2: Within-SG shuffle (from coalesced within-SG layout).
// Phase 3: Cross-SG via shared memory, second local reduce.
// ============================================================================

#layout_complex_4d = #iree_map.pack_layout<((2,2,2), (4,2), (2,2), (4,4)) : ((0,1,0), (2,128), (8,64), (16,0))>

// CHECK-LABEL: @reduce_4d_multi_dim_cross_subgroup
//   Local reduce dims 1,3 value leaves:
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2, 4] : vector<2x2x1x1x4xf16> to vector<2x2x1xf16>
//   Within-SG shuffle: two clusters per element (multi-mode within-SG grid).
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 2, stride = 16) : (f16) -> f16
// CHECK:       gpu.subgroup_reduce add %{{.*}} cluster(size = 4, stride = 2) : (f16) -> f16
//   Shared memory buffer [8, 4, 2, 2] (parallel dims + across-SG modes):
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<8x4x2x2xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Write partial results:
// CHECK:       vector.transfer_write {{.*}}, %[[ALLOC]]{{.*}} : vector<1xf16>, memref<8x4x2x2xf16, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
//   Read all partial results:
// CHECK:       vector.transfer_read %[[ALLOC]]{{.*}} : memref<8x4x2x2xf16, #gpu.address_space<workgroup>>, vector<2xf16>
//   Second reduction (over across-SG dims):
// CHECK:       vector.multi_reduction <add>, %{{.*}}, %{{.*}} [3, 4] : vector<2x2x1x2x2xf16> to vector<2x2x1xf16>
//   Combine with accumulator:
// CHECK:       arith.addf %{{.*}}, %{{.*}} : vector<2x2x1xf16>
func.func @reduce_4d_multi_dim_cross_subgroup(%arg0: memref<8x8x4x16xf16>, %acc: vector<8x4xf16>) -> vector<8x4xf16>
    attributes { workgroup_size = [256 : index] } {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %v = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]}
    : memref<8x8x4x16xf16>, vector<8x8x4x16xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_complex_4d)
    : vector<8x8x4x16xf16>
  %r = vector.multi_reduction <add>, %vl, %acc [1, 3]
    : vector<8x8x4x16xf16> to vector<8x4xf16>
  func.return %r : vector<8x4xf16>
}
