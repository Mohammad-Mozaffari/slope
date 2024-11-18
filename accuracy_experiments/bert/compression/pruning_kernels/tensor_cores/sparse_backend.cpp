#include <torch/extension.h>
#include <cusparseLt.h>       // cusparseLt header
#include <iostream>

#define CHECK_CUDA_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_DEVICE(x); CHECK_CONTIGUOUS(x)

int init_cusparse_lt_cuda();
int setup_prune_matmul( const int                       m,
                        const int                       n,
                        const int                       k,
                        at::Half                        *dSparse,
                        at::Half                        *dDense,
                        int                             *index,
                        const bool                      transpose_A=false,
                        const bool                      transpose_B=false,
                        const bool                      sparseA=true,
                        const bool                      transposable_mask=false);


torch::Tensor spmatmul_cuda(at::Half    *dDense,
                            int         index,
                            bool        sparseA);


void save_grad_cuda(torch::Tensor grad, int index);


torch::Tensor init_cusparse_lt() {
  int result = init_cusparse_lt_cuda();
  if(result == EXIT_SUCCESS) {
    return torch::zeros({1}, torch::kInt32);
  } else {
    return torch::ones({1}, torch::kInt32);
  }
}


torch::Tensor setup_spmatmul(torch::Tensor A,
                                torch::Tensor B,
                                const bool transpose_A=false,
                                const bool transpose_B=false,
                                const bool sparseA=true,
                                const bool transposable_mask=false) {

   CHECK_INPUT(A);
   CHECK_INPUT(B);
   auto index = torch::zeros({1}, torch::kInt32);
   int result;
   auto sparse_mat = sparseA ? A.data_ptr<at::Half>() : B.data_ptr<at::Half>();
   auto dense_mat = sparseA ? B.data_ptr<at::Half>() : A.data_ptr<at::Half>();
   at::Half *dCompressed;
   int m, k, n;
   if(transpose_A && transpose_B)
   {
        m = A.size(1);
        k = A.size(0);
        n = B.size(0);
   } else if(transpose_A)
   {
        m = A.size(1);
        k = A.size(0);
        n = B.size(1);
   } else if(transpose_B)
   {
        m = A.size(0);
        k = A.size(1);
        n = B.size(0);
   } else {
        m = A.size(0);
        k = A.size(1);
        n = B.size(1);
   }
   result = setup_prune_matmul(     m,
                                    n,
                                    k,
                                    sparse_mat,
                                    dense_mat,
                                    index.data_ptr<int>(),
                                    transpose_A,
                                    transpose_B,
                                    sparseA,
                                    transposable_mask);
   if(result == EXIT_SUCCESS) {
     return index;
   } else {
     return -torch::ones({1}, torch::kInt32);
   }
}


torch::Tensor spmatmul( torch::Tensor Dense,
                        torch::Tensor index,
                        const bool sparseA=true) {
   CHECK_INPUT(Dense);
//   std::cout << Dense.data_ptr<at::Half>()[0] << std::endl;
   auto result = spmatmul_cuda(     Dense.data_ptr<at::Half>(),
                                    *index.data_ptr<int>(),
                                    sparseA);
   return result;
}


torch::Tensor save_grad(torch::Tensor input, torch::Tensor index) {
    CHECK_INPUT(input);
    save_grad_cuda(input, *index.data_ptr<int>());
}


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


torch::Tensor sparse_add_cuda(torch::Tensor dense, torch::Tensor sparse_index, torch::Tensor alpha, torch::Tensor beta);


torch::Tensor sparse_add(
        torch::Tensor dense, torch::Tensor sparse_index, torch::Tensor alpha, torch::Tensor beta) {
    CHECK_INPUT(dense);
    return sparse_add_cuda(dense, sparse_index, alpha, beta);
}


void update_sparse_matrix_cuda(torch::Tensor new_data, torch::Tensor sparse_idx);


void update_sparse_matrix(
        torch::Tensor new_data, torch::Tensor sparse_idx) {
    CHECK_INPUT(new_data);
    update_sparse_matrix_cuda(new_data, sparse_idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_cusparse_lt", &init_cusparse_lt, "Initialize CUSPARSE LT");
    m.def("setup_spmatmul", &setup_spmatmul, "Setup Sparse Matrix Multiplication");
    m.def("spmatmul", &spmatmul, "Sparse Matrix Multiplication");
    m.def("save_grad", &save_grad, "Save Gradient");
    m.def("prune", &prune, "N:M Prune (CUDA)");
    m.def("prune_and_compress", &prune_and_compress, "Prune the dense matrix using the mask and store it in a "
                                                     "compressed tensor (CUDA)");
    m.def("sparse_add", &sparse_add, "Add the sparse matrix to the dense matrix and return a "
                                     "compressed dense matrix(CUDA)");
    m.def("update_sparse_matrix", &update_sparse_matrix, "Update the sparse matrix with the new dense matrix "
                                                         "data (CUDA)");
}