#sync = #iree_linalg_ext.requires_sync<#gpu.address_space<workgroup>, 1>
#sync2 = #iree_linalg_ext.requires_sync<#gpu.address_space<workgroup>, 2>

func.func @test(%mem: memref<128xf32>) {
  %shared = memref.alloc() : memref<1xf32, #sync>
  %shared2 = memref.alloc() : memref<1xf32, #sync2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %compute_0 = arith.constant dense<0.0> : vector<1xf32>
  %a_0 = vector.transfer_read %mem[%c0], %cst_0 : memref<128xf32>, vector<1xf32>
  vector.transfer_write %a_0, %shared[%c0] : vector<1xf32>, memref<1xf32, #sync>
  %compute_n = scf.for %i = %c1 to %c128 step %c2 iter_args(%compute_i = %compute_0) -> (vector<1xf32>) {

    // Fetch i
    %a_i = vector.transfer_read %mem[%i], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %a_i, %shared2[%c0] : vector<1xf32>, memref<1xf32, #sync2>

    // Compute i - 1
    %ci = vector.transfer_read %shared[%c0], %cst_0 : memref<1xf32, #sync>, vector<1xf32>
    %computei1 = arith.addf %compute_i, %ci : vector<1xf32>

    // Fetch i + 1
    %ip1 = arith.addi %i, %c1 : index
    %a_i2 = vector.transfer_read %mem[%ip1], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %a_i2, %shared[%c0] : vector<1xf32>, memref<1xf32, #sync>
    
    // Compute i
    %ci2 = vector.transfer_read %shared2[%c0], %cst_0 : memref<1xf32, #sync2>, vector<1xf32>
    %computei2 = arith.addf %computei1, %ci2 : vector<1xf32>

    scf.yield %computei2 : vector<1xf32>
  }
  %c = vector.transfer_read %shared[%c0], %cst_0 : memref<1xf32, #sync>, vector<1xf32>
  %compute = arith.addf %compute_n, %c : vector<1xf32>
  vector.transfer_write %compute, %mem[%c0] : vector<1xf32>, memref<128xf32>
  func.return
}
