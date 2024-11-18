#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT8(x) TORCH_CHECK(x.dtype() == torch::kInt8, "Input tensor 'a' must have dtype int8");
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x);
void int8_matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

void int_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

   if (A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8) {
         int8_matmul_cuda(A, B, C);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int_matmul", &int_matmul, "Integer Matrix Matrix Multiplication (cuda)");
}