// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-tensor-layouts{use-pack-layouts=true}, canonicalize, cse))' %s | FileCheck %s

// ============================================================================
// Contraction: MFMA_F32_32x32x8_F16, subgroupSize=64
// ============================================================================

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_mfma(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$LHS:.+]] = #iree_map.pack_layout<((3, 32), (2, 2, 4)) : ((0, 1), (0, 32, 0))>
// CHECK-DAG: #[[$RHS:.+]] = #iree_map.pack_layout<((2, 32), (2, 2, 4)) : ((0, 1), (0, 32, 0))>
// CHECK-DAG: #[[$ACC:.+]] = #iree_map.pack_layout<((12, 2, 4), (2, 32)) : ((0, 32, 0), (0, 1))>

// CHECK-LABEL: func.func @matmul_96x64x16_mfma

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$LHS]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$RHS]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$ACC]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

// ============================================================================
// Contraction: WMMAR3_F32_16x16x16_F16, subgroupSize=64
// ============================================================================

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar3(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$LHS:.+]] = #iree_map.pack_layout<((6, 16), 16) : ((0, 1), 0)>
// CHECK-DAG: #[[$RHS:.+]] = #iree_map.pack_layout<((4, 16), 16) : ((0, 1), 0)>
// CHECK-DAG: #[[$ACC:.+]] = #iree_map.pack_layout<((48, 2), (4, 16)) : ((0, 16), (0, 1))>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar3

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$LHS]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$RHS]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$ACC]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

// ============================================================================
// Non-contraction: general GPU config with subgroup + thread basis
// ============================================================================

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 2, 1]
                                              subgroup_size = 64>

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @elementwise_128x256(%A: tensor<128x256xf16>, %B: tensor<128x256xf16>)
    -> tensor<128x256xf16>
    attributes { translation_info = #translation } {
  %out = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"],
    lowering_config = #iree_gpu.lowering_config<{
      subgroup_basis = [[2, 1], [0, 1]],
      thread = [1, 4],
      lane_basis = [[64, 1], [0, 1]]
    }>
  } ins(%A, %B : tensor<128x256xf16>, tensor<128x256xf16>)
    outs(%A : tensor<128x256xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %add = arith.addf %a, %b : f16
    linalg.yield %add : f16
  } -> tensor<128x256xf16>
  return %out : tensor<128x256xf16>
}

// Subgroup (2, stride=64) and thread (64, stride=1) on dim 0 coalesce to (128, 1).
// Batch (64) and element (4) on dim 1 coalesce to 256:0.
// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_map.pack_layout<(128, 256) : (1, 0)>

// CHECK-LABEL: func.func @elementwise_128x256
// CHECK: iree_vector_ext.to_layout %{{.*}} to layout(#[[$LAYOUT]])
