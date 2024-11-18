#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
//#define CHECK_INPUT(x) CHECK_CUDA(x);

void get_half_min_max_cuda(torch::Tensor mat, torch::Tensor mins, torch::Tensor maxs);
void get_float_min_max_cuda(torch::Tensor mat, torch::Tensor mins, torch::Tensor maxs);

void get_min_max(torch::Tensor mat, torch::Tensor mins, torch::Tensor maxs) {
  CHECK_INPUT(mat);
  CHECK_INPUT(mins);
  CHECK_INPUT(maxs);
  if (mat.dtype() == torch::kFloat16)
        get_half_min_max_cuda(mat, mins, maxs);
  else
        get_float_min_max_cuda(mat, mins, maxs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_min_max", &get_min_max, "Extracts row-wise minimum and maximum of a matrix (cuda)");
}