// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @to_simt_to_simd_fold
// CHECK-SAME: (%[[SIMD:.*]]: vector<64x64xf32>) -> vector<64x64xf32>
func.func @to_simt_to_simd_fold(%simd: vector<64x64xf32>) -> vector<64x64xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simd_out = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK: return %[[SIMD]]
  func.return %simd_out : vector<64x64xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_fold
// CHECK-SAME: (%[[SIMT:.*]]: vector<4x4x4xf32>) -> vector<4x4x4xf32>
func.func @to_simd_to_simt_fold(%simt: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK: return %[[SIMT]]
  func.return %simt_out : vector<4x4x4xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_multi_use
// CHECK-SAME: (%[[SIMT:.*]]: vector<4x4x4xf32>)
func.func @to_simd_to_simt_multi_use(%simt: vector<4x4x4xf32>) -> (vector<4x4x4xf16>, vector<64x64xf32>) {
  // The to_simd operation should not be dce-ed after folding because it is returned.
  // CHECK: %[[SIMD:.*]] = iree_vector_ext.to_simd %[[SIMT]] : vector<4x4x4xf32> -> vector<64x64xf32>
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // The to_simt operation should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>

  // Check if the folding happened correctly.
  // CHECK: %[[TRUNCED:.*]] = arith.truncf %[[SIMT]]
  %trunced = arith.truncf %simt_out : vector<4x4x4xf32> to vector<4x4x4xf16>

  // CHECK: return %[[TRUNCED]], %[[SIMD]]
  func.return %trunced, %simd : vector<4x4x4xf16>, vector<64x64xf32>
}

// -----

func.func @transfer_gather_fold_broadcast(%indices: vector<64xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %broadcasted = vector.broadcast %indices : vector<64xindex> to vector<32x64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%broadcasted : vector<32x64xindex>], %cst0 {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1)[s0] -> (d1, d0)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$SMAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$IVMAP:.*]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @transfer_gather_fold_broadcast
// CHECK: transfer_gather
// CHECK-SAME: indexing_maps = [#[[$SMAP]], #[[$IVMAP]]]

// -----

func.func @transfer_gather_fold_transpose(%indices: vector<64x32xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %transposed = vector.transpose %indices, [1, 0] : vector<64x32xindex> to vector<32x64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%transposed : vector<32x64xindex>], %cst0 {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1)[s0] -> (d1, d0)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$SMAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$IVMAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @transfer_gather_fold_transpose
// CHECK: transfer_gather
// CHECK-SAME: indexing_maps = [#[[$SMAP]], #[[$IVMAP]]]

// -----

func.func @transfer_gather_fold_step(%indices: vector<64x32xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %step = vector.step : vector<64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%step, %indices : vector<64xindex>, vector<64x32xindex>], %cst0 {
    indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                     affine_map<(d0, d1)[s0, s1] -> (d0)>,
                     affine_map<(d0, d1)[s0, s1] -> (d0, d1)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-LABEL: @transfer_gather_fold_step
// CHECK-SAME: %[[ARG1:.*]]: vector<64x32xindex>
// CHECK: transfer_gather
// CHECK-SAME: [%[[ARG1]] : vector<64x32xindex>]

// -----

func.func @transfer_gather_fold_single_element(%scalar: vector<1xindex>,
  %indices: vector<64x1xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x1xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%scalar, %indices : vector<1xindex>, vector<64x1xindex>], %cst0 {
    indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                     affine_map<(d0, d1)[s0, s1] -> (d1)>,
                     affine_map<(d0, d1)[s0, s1] -> (d0, d1)>]
  } : tensor<4096x64xf16>, vector<64x1xf16>

  return %out : vector<64x1xf16>
}

// CHECK-LABEL: @transfer_gather_fold_single_element
// CHECK-SAME: %{{.*}}: vector<1xindex>, %[[ARG1:.*]]: vector<64x1xindex>
// CHECK: transfer_gather
// CHECK-SAME: [%[[ARG1]] : vector<64x1xindex>]

// -----

func.func @transfer_gather_fold_add_broadcast(%indices: vector<64xindex>,
  %source: tensor<4096x64xf16>, %offset: index)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %bcast = vector.broadcast %offset : index to vector<64xindex>
  %added = arith.addi %indices, %bcast : vector<64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%added : vector<64xindex>], %cst0 {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-LABEL: @transfer_gather_fold_add_broadcast
// CHECK-SAME: %[[INDICES:.*]]: vector<64xindex>, %[[SOURCE:.*]]: tensor<4096x64xf16>, %[[OFFSET:.*]]: index
// CHECK: transfer_gather %[[SOURCE]][%[[OFFSET]],
// CHECK-SAME: [%[[INDICES]] : vector<64xindex>]

// -----

func.func @transfer_gather_fold_contiguous_load(
  %source: tensor<4096x64xf16>)
  -> vector<64x1xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0], %cst0 {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>]
  } : tensor<4096x64xf16>, vector<64x1xf16>

  return %out : vector<64x1xf16>
}

// CHECK-LABEL: @transfer_gather_fold_contiguous_load
// CHECK: vector.transfer_read
// CHECK-NOT: transfer_gather

// -----

func.func @transfer_gather_fold_all_true_mask(
  %source: tensor<4096x64xf16>, %indices: vector<64xindex>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %mask = arith.constant dense<true> : vector<64x32xi1>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<64xindex>], %cst0, %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>, vector<64x32xi1>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$SMAP:.*]] = affine_map<(d0, d1)[s0] -> (s0, d1)>
// CHECK-DAG: #[[$IVMAP:.*]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @transfer_gather_fold_all_true_mask
// CHECK: iree_vector_ext.transfer_gather
// CHECK-SAME: indexing_maps = [#[[$SMAP]], #[[$IVMAP]]]
// CHECK-SAME: : tensor<4096x64xf16>, vector<64x32xf16>
// CHECK-NOT: vector<64x32xi1>

// -----

func.func @transfer_gather_fold_all_false_mask(
  %source: tensor<4096x64xf16>, %indices: vector<64xindex>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %mask = arith.constant dense<false> : vector<64x32xi1>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<64xindex>], %cst0, %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : tensor<4096x64xf16>, vector<64x32xf16>, vector<64x32xi1>

  return %out : vector<64x32xf16>
}

// CHECK-LABEL: @transfer_gather_fold_all_false_mask
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<64x32xf16>
// CHECK: return %[[CST]]
// CHECK-NOT: transfer_gather

// -----

// CHECK-LABEL: @fold_zero_dim_reduce
// CHECK-SAME: (%[[INPUT:.*]]: vector<4x64xf32>)
func.func @fold_zero_dim_reduce(%input: vector<4x64xf32>) -> vector<4x64xf32> {
  // CHECK-NOT: iree_vector_ext.associative_reduce
  %0 = iree_vector_ext.associative_reduce ins(%input : vector<4x64xf32>) [] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<4x64xf32>
  // CHECK: return %[[INPUT]]
  return %0 : vector<4x64xf32>
}

// -----

// CHECK-LABEL: @fold_zero_dim_reduce_multi
// CHECK-SAME: (%[[IN0:.*]]: vector<4x64xf32>, %[[IN1:.*]]: vector<4x64xi32>)
func.func @fold_zero_dim_reduce_multi(%in0: vector<4x64xf32>, %in1: vector<4x64xi32>) -> (vector<4x64xf32>, vector<4x64xi32>) {
  // CHECK-NOT: iree_vector_ext.associative_reduce
  %0:2 = iree_vector_ext.associative_reduce ins(%in0, %in1 : vector<4x64xf32>, vector<4x64xi32>) [] {
  ^bb0(%a0: f32, %a1: i32, %b0: f32, %b1: i32):
    %cmp = arith.cmpf ogt, %a0, %b0 : f32
    %sel0 = arith.select %cmp, %a0, %b0 : f32
    %sel1 = arith.select %cmp, %a1, %b1 : i32
    iree_vector_ext.yield %sel0, %sel1 : f32, i32
  } -> vector<4x64xf32>, vector<4x64xi32>
  // CHECK: return %[[IN0]], %[[IN1]]
  return %0#0, %0#1 : vector<4x64xf32>, vector<4x64xi32>
}

// -----

// CHECK-LABEL: @fold_zero_dim_scan
// CHECK-SAME: (%[[INPUT:.*]]: vector<4x64xf32>)
func.func @fold_zero_dim_scan(%input: vector<4x64xf32>) -> vector<4x64xf32> {
  // CHECK-NOT: iree_vector_ext.associative_scan
  %0 = iree_vector_ext.associative_scan ins(%input : vector<4x64xf32>) [] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<4x64xf32>
  // CHECK: return %[[INPUT]]
  return %0 : vector<4x64xf32>
}

// -----

// CHECK-LABEL: @inline_small_reduce_size2
// CHECK-SAME: (%[[INPUT:.*]]: vector<4x2xf32>)
func.func @inline_small_reduce_size2(%input: vector<4x2xf32>) -> vector<4xf32> {
  // CHECK-NOT: iree_vector_ext.associative_reduce
  // CHECK: arith.addf {{.*}} : vector<4xf32>
  %0 = iree_vector_ext.associative_reduce ins(%input : vector<4x2xf32>) [1] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @inline_small_reduce_argmax
// CHECK-SAME: (%[[IN0:.*]]: vector<4x2xf32>, %[[IN1:.*]]: vector<4x2xi32>)
func.func @inline_small_reduce_argmax(%in0: vector<4x2xf32>, %in1: vector<4x2xi32>) -> (vector<4xf32>, vector<4xi32>) {
  // CHECK-NOT: iree_vector_ext.associative_reduce
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  // CHECK: arith.select
  %0:2 = iree_vector_ext.associative_reduce ins(%in0, %in1 : vector<4x2xf32>, vector<4x2xi32>) [1] {
  ^bb0(%a0: f32, %a1: i32, %b0: f32, %b1: i32):
    %cmp = arith.cmpf ogt, %a0, %b0 : f32
    %sel0 = arith.select %cmp, %a0, %b0 : f32
    %sel1 = arith.select %cmp, %a1, %b1 : i32
    iree_vector_ext.yield %sel0, %sel1 : f32, i32
  } -> vector<4xf32>, vector<4xi32>
  return %0#0, %0#1 : vector<4xf32>, vector<4xi32>
}

// -----

// Verify that large reductions are NOT inlined.
// CHECK-LABEL: @no_inline_large_reduce
func.func @no_inline_large_reduce(%input: vector<4x64xf32>) -> vector<4xf32> {
  // CHECK: iree_vector_ext.associative_reduce
  %0 = iree_vector_ext.associative_reduce ins(%input : vector<4x64xf32>) [1] {
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    iree_vector_ext.yield %add : f32
  } -> vector<4xf32>
  return %0 : vector<4xf32>
}
