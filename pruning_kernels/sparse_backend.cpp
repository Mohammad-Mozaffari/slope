#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_DEVICE(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> prune_cuda(torch::Tensor input, const int N, const int M);


std::vector<torch::Tensor> prune(
        torch::Tensor input, const int N, const int M) {
    CHECK_INPUT(input);
    return prune_cuda(input, N, M);
}


torch::Tensor prune_and_compress_cuda(torch::Tensor input, torch::Tensor mask);


torch::Tensor prune_and_compress(
        torch::Tensor input, torch::Tensor mask) {
    CHECK_INPUT(input);
    return prune_and_compress_cuda(input, mask);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prune", &prune, "N:M Prune (CUDA)");
    m.def("prune_and_compress", &prune_and_compress, "Prune the dense matrix using the mask and store it in a "
                                                     "compressed tensor (CUDA)");
}
