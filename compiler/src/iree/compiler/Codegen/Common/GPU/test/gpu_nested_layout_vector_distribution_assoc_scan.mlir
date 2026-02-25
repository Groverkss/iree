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

func.func @assoc_scan_addf_dim1(%arg0: vector<32x32xf32>) -> vector<32x32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = iree_vector_ext.associative_scan ins(%arg0l : vector<32x32xf32>) [1] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<32x32xf32>
  return %0 : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Dim 1 layout: batch=2, outer=1, thread=4, element=4 -> merged local scan dim = 8
// Thread scan: thread_tile[1]=4, stride=16, width=64 -> 2 Hillis-Steele rounds
// CHECK-LABEL: func @assoc_scan_addf_dim1
// CHECK-DAG: %[[DARG0:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf32> -> vector<2x2x1x1x1x4xf32>
//     Local scan: shape_cast to merge scan tiers, associative_scan, shape_cast back
// CHECK: vector.shape_cast %[[DARG0]] : vector<2x2x1x1x1x4xf32> to vector<2x1x1x8xf32>
// CHECK: iree_vector_ext.associative_scan
// CHECK: vector.shape_cast {{.*}} : vector<2x1x1x8xf32> to vector<2x2x1x1x1x4xf32>
//     Thread scan: Hillis-Steele shuffle up
// CHECK: gpu.shuffle up %{{.*}}, %{{.*}}, %{{.*}} : f32
// CHECK: arith.addf
// CHECK: arith.select
//     Exclusive prefix broadcast and combine
// CHECK: gpu.shuffle up
// CHECK: vector.broadcast
// CHECK: arith.addf {{.*}} : vector<2x2x1x1x1x4xf32>
// CHECK: arith.select
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x2x1x1x1x4xf32> -> vector<32x32xf32>

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

func.func @assoc_scan_maximumf_dim0(%arg0: vector<32x32xf32>) -> vector<32x32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = iree_vector_ext.associative_scan ins(%arg0l : vector<32x32xf32>) [0] {
  ^bb0(%a: f32, %b: f32):
    %max = arith.maximumf %a, %b : f32
    iree_vector_ext.yield %max : f32
  } -> vector<32x32xf32>
  return %0 : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Dim 0 layout: batch=2, outer=1, thread=16, element=1 -> merged local scan dim = 2
// Thread scan: thread_tile[0]=16, stride=1, width=16 -> 4 Hillis-Steele rounds
// CHECK-LABEL: func @assoc_scan_maximumf_dim0
//     Local scan: transpose to group scan tiers at end, merge, scan, split, transpose back
// CHECK: vector.transpose
// CHECK: vector.shape_cast
// CHECK: iree_vector_ext.associative_scan
// CHECK: vector.shape_cast
// CHECK: vector.transpose
//     Thread scan: shuffle up rounds
// CHECK: gpu.shuffle up %{{.*}}, %{{.*}}, %{{.*}} : f32
// CHECK: arith.maximumf
// CHECK: arith.select
//     Exclusive prefix broadcast and combine
// CHECK: gpu.shuffle up
// CHECK: vector.broadcast
// CHECK: arith.maximumf {{.*}} : vector<2x2x1x1x1x4xf32>
// CHECK: arith.select
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x2x1x1x1x4xf32> -> vector<32x32xf32>
