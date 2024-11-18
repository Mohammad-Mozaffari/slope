#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>



void int8_matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

    cublasHandle_t handle;
    cublasCreate(&handle);
//
    int row_a = A.sizes()[0];
    int col_a = A.sizes()[1];
    int row_b = B.sizes()[0];
    int col_b = B.sizes()[1];

    int alpha = 1;
    int beta = 0;

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, row_a, row_b, col_a,
                  &alpha, A.data_ptr<int8_t>(), CUDA_R_8I, col_a, B.data_ptr<int8_t>(), CUDA_R_8I, col_b, &beta,
                  C.data_ptr<int32_t>(), CUDA_R_32I, row_a, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);

}
