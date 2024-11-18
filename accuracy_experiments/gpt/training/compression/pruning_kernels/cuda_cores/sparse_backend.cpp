#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> prune_cuda(torch::Tensor input, const int N, const int M);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> prune(
    torch::Tensor input, const int N, const int M) {
  CHECK_INPUT(input);
  return prune_cuda(input, N, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prune", &prune, "N:M Prune (CUDA)");
}