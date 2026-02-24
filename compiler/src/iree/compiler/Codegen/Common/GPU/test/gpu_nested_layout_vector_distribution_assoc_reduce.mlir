// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize -mlir-print-local-scope --cse %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @assoc_reduce_addf_dim1(%arg0: vector<32x32xf32>) -> vector<32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = iree_vector_ext.associative_reduce ins(%arg0l : vector<32x32xf32>) [1] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @assoc_reduce_addf_dim1
// CHECK-DAG: %[[DARG0:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf32> -> vector<2x2x1x1x1x4xf32>
// Local reduction: matched to vector.multi_reduction with identity init
// CHECK: vector.multi_reduction <add>, %[[DARG0]], %{{.*}} [1, 3, 5] : vector<2x2x1x1x1x4xf32> to vector<2x1x1xf32>
// Thread reduction: gpu.subgroup_reduce with cluster_size=4, stride=16
// CHECK: gpu.subgroup_reduce add %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Result converted back to SIMD
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x1x1xf32> -> vector<32xf32>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @assoc_reduce_maximumf_dim0(%arg0: vector<32x32xf32>) -> vector<32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = iree_vector_ext.associative_reduce ins(%arg0l : vector<32x32xf32>) [0] {
  ^bb0(%a: f32, %b: f32):
    %max = arith.maximumf %a, %b : f32
    iree_vector_ext.yield %max : f32
  } -> vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @assoc_reduce_maximumf_dim0
// Local reduction with maximumf identity
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [0, 2, 4] : vector<2x2x1x1x1x4xf32> to vector<2x1x4xf32>
// Thread reduction: dim 0 has 16 threads
// CHECK: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 16) : (f32) -> f32
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x1x4xf32> -> vector<32xf32>
