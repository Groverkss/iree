// RUN: iree-opt --pass-pipeline="builtin.module(any(iree-codegen-gpu-map-vector-distribution),canonicalize,cse)" --split-input-file %s | FileCheck %s

// ============================================================================
// InnerTiled MMA: MFMA_F32_16x16x16_F16 with outer iterations (i=2, j=4, k=3)
// 64 threads total. Each thread holds 4 elements per MMA tile operand.
//
// LHS (i,k) = 2x3x16x16: inner M has 16 threads, inner K has 4 val + 4 thread
// RHS (k,j) = 3x4x16x16: inner K has 4 val + 4 thread, inner N has 16 threads
// ACC (i,j) = 2x4x16x16: inner M has 4 val + 4 thread, inner N has 16 threads
// ============================================================================

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]

#lhs_layout = #iree_map.pack_layout<(2, 3, 16, (4, 4)) : (0, 0, 1, (0, 16))>
#rhs_layout = #iree_map.pack_layout<(3, 4, (4, 4), 16) : (0, 0, (0, 16), 1)>
#acc_layout = #iree_map.pack_layout<(2, 4, (4, 4), 16) : (0, 0, (0, 16), 1)>

// CHECK-LABEL: @inner_tiled_mma
// CHECK-DAG:   %[[LHS_SIMT:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<2x3x16x16xf16> -> vector<2x3x1x4xf16>
// CHECK-DAG:   %[[RHS_SIMT:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<3x4x16x16xf16> -> vector<3x4x4x1xf16>
// CHECK-DAG:   %[[ACC_SIMT:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<2x4x16x16xf32> -> vector<2x4x4x1xf32>
// CHECK:       %[[RESULT:.*]] = iree_codegen.inner_tiled ins(%[[LHS_SIMT]], %[[RHS_SIMT]]) outs(%[[ACC_SIMT]])
// CHECK-SAME:    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
// CHECK-SAME:    : vector<2x3x1x4xf16>, vector<3x4x4x1xf16> into vector<2x4x4x1xf32>
// CHECK:       iree_vector_ext.to_simd %[[RESULT]] : vector<2x4x4x1xf32> -> vector<2x4x16x16xf32>
func.func @inner_tiled_mma(
    %lhs: vector<2x3x16x16xf16>,
    %rhs: vector<3x4x16x16xf16>,
    %acc: vector<2x4x16x16xf32>) -> vector<2x4x16x16xf32>
    attributes { workgroup_size = [64 : index] } {
  %A = iree_vector_ext.to_layout %lhs to layout(#lhs_layout) : vector<2x3x16x16xf16>
  %B = iree_vector_ext.to_layout %rhs to layout(#rhs_layout) : vector<3x4x16x16xf16>
  %C = iree_vector_ext.to_layout %acc to layout(#acc_layout) : vector<2x4x16x16xf32>

  %result = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : vector<2x3x16x16xf16>, vector<3x4x16x16xf16> into vector<2x4x16x16xf32>

  %O = iree_vector_ext.to_layout %result to layout(#acc_layout) : vector<2x4x16x16xf32>
  return %O : vector<2x4x16x16xf32>
}
