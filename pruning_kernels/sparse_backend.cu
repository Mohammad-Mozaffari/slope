#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <vector>             // std::vector
#include <torch/extension.h>
#include <iostream>


#define INT8_OUTPUT_TYPE int8_t //at::Half //int8_t
#define INT8_OUTPUT_TYPE_CUDA CUDA_R_8I //CUDA_R_32I
#define INT8_OUTPUT_TYPE_TORCH torch::kInt8 //torch::kInt32


#define MAX(a, b) ((abs(a) > abs(b) ? (a) : (b)))
#define MIN(a, b) ((abs(a) < abs(b) ? (a) : (b)))


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


#define CHECK_CUDA_TORCH(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return torch::ones(1);                                                   \
    }                                                                          \
}


__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        if(abs(output[index]) > abs(output[index + 1])){
            output[index + 1] = 0.;
            mask[index + 1] = true;
        } else {
            output[index] = 0.;
            mask[index] = true;
        }
        if(abs(output[index + 2]) > abs(output[index + 3])){
            output[index + 3] = 0.;
            mask[index + 3] = true;
        } else {
            output[index + 2] = 0.;
            mask[index + 2] = true;
        }
  }
}


template <class T>
__global__ void prune_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        T min1, min2;
        int min_idx1, min_idx2;
        min1 = output[index];
        min_idx1 = index;
        if(MIN(min1, output[index + 1]) == output[index + 1]){
            min1 = output[index + 1];
            min_idx1 = index + 1;
        }
        if(MIN(min1, output[index + 2]) == output[index + 2]){
            min1 = output[index + 2];
            min_idx1 = index + 2;
        }
        if(MIN(min1, output[index + 3]) == output[index + 3]){
            min1 = output[index + 3];
            min_idx1 = index + 3;
        }
        min2 = min_idx1 == index ? output[index + 1] : output[index];
        min_idx2 = min_idx1 == index ? index + 1 : index;
        if((MIN(min2, output[index + 1]) == output[index + 1]) && min_idx1 != index + 1){
            min2 = output[index + 1];
            min_idx2 = index + 1;
        }
        if((MIN(min2, output[index + 2]) == output[index + 2]) && min_idx1 != index + 2){
            min2 = output[index + 2];
            min_idx2 = index + 2;
        }
        if((MIN(min2, output[index + 3]) == output[index + 3]) && min_idx1 != index + 3){
            min2 = output[index + 3];
            min_idx2 = index + 3;
        }
        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;

        min1 = output[index + 4];
        min_idx1 = index + 4;
        if(MIN(min1, output[index + 5]) == output[index + 5]){
            min1 = output[index + 5];
            min_idx1 = index + 5;
        }
        if(MIN(min1, output[index + 6]) == output[index + 6]){
            min1 = output[index + 6];
            min_idx1 = index + 6;
        }
        if(MIN(min1, output[index + 7]) == output[index + 7]){
            min1 = output[index + 7];
            min_idx1 = index + 7;
        }
        min2 = min_idx1 == index + 4 ? output[index + 5] : output[index + 4];
        min_idx2 = min_idx1 == index + 4 ? index + 5 : index + 4;
        if((MIN(min2, output[index + 5]) == output[index + 5]) && min_idx1 != index + 5){
            min2 = output[index + 5];
            min_idx2 = index + 5;
        }
        if((MIN(min2, output[index + 6]) == output[index + 6]) && min_idx1 != index + 6){
            min2 = output[index + 6];
            min_idx2 = index + 6;
        }
        if((MIN(min2, output[index + 7]) == output[index + 7]) && min_idx1 != index + 7){
            min2 = output[index + 7];
            min_idx2 = index + 7;
        }

        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;
  }
}


template <class T>
__device__ void find_kth_smallest(
                                    int *smallest_idx,
                                    const T* __restrict__ input,
                                    const int k,
                                    const int M, int index) {
    int min_idx = 0;
    T min = 6.0e4;

    for(int i = 0; i < M; i++)
    {
        bool ignore = false;
        for(int j = 0; j < k; j++)
        {
            if(smallest_idx[j] == i)
            {
                ignore = true;
            }
        }
        if(ignore)
        {
            continue;
        }
        if(MIN(min, input[i]) == input[i]){
            min = input[i];
            min_idx = i;
        }
    }
    smallest_idx[k] = min_idx;
}


template <class T>
__global__ void prune_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size,
        const int N,
        const int M) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 8; i++)
        {
            reinterpret_cast<float4*>(&output[index + 8 * i])[0] = reinterpret_cast<const float4*>(&input[index + 8 * i])[0];
        }

        int min_idx_list[16];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<T>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size,
        const int N,
        const int M) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 4; i++)
        {
            reinterpret_cast<float4*>(&output[index + 4 * i])[0] = reinterpret_cast<const float4*>(&input[index + 4 * i])[0];
        }

        int *min_idx_list;
        min_idx_list = (int*)malloc((M - N) * sizeof(int));
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<float>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


template <int N, int M>
__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 4; i++)
        {
            reinterpret_cast<float4*>(&output[index + 4 * i])[0] = reinterpret_cast<const float4*>(&input[index + 4 * i])[0];
        }

        int min_idx_list[M - N];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<float>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


template <int N, int M, class T>
__global__ void prune_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 8; i++)
        {
            reinterpret_cast<float4*>(&output[index + 8 * i])[0] = reinterpret_cast<const float4*>(&input[index + 8 * i])[0];
        }

        int min_idx_list[M - N];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<T>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


std::vector<torch::Tensor> prune_cuda(
    torch::Tensor input, const int N, const int M) {

    auto output = torch::zeros_like(input);
    auto options = torch::TensorOptions().dtype(torch::kBool);
    auto mask = torch::zeros_like(input, options);

    const auto batch_size = input.size(0);
    const auto row_size = input.size(1);

    const int threads = 1024;

    if(N == 1 && M == 2) {
        switch (input.type().scalarType()) {
            case torch::ScalarType::Float: {
                const dim3 blocks(((row_size / 4) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                        input.data<float>(),
                        output.data<float>(),
                        mask.data<bool>(),
                        row_size);
                break;
            }
            case torch::ScalarType::Half: {
                throw std::runtime_error("Half precision not supported for N=1, M=2");
            }
            case torch::ScalarType::BFloat16: {
                throw std::runtime_error("BFloat16 precision not supported for N=1, M=2");
            }
            default: {
                throw std::runtime_error("Unsupported data type");
            }
        }
    }
    else if(N == 2 && M == 4)
    {
            switch (input.type().scalarType()) {
                case torch::ScalarType::Float: {
                    throw std::runtime_error("Full precision not supported for N=2, M=4");
                    break;
                }
                case torch::ScalarType::Half: {
                    const dim3 blocks(((row_size / 8) + threads - 1) / threads, batch_size);
                    prune_kernel<at::Half><<<blocks, threads>>>(
                            input.data<at::Half>(),
                            output.data<at::Half>(),
                            mask.data<bool>(),
                            row_size);
                    break;
                }
                case torch::ScalarType::BFloat16: {
                    const dim3 blocks(((row_size / 8) + threads - 1) / threads, batch_size);
                    prune_kernel<at::BFloat16><<<blocks, threads>>>(
                            input.data<at::BFloat16>(),
                            output.data<at::BFloat16>(),
                            mask.data<bool>(),
                            row_size);
                    break;
                }
                default: {
                    throw std::runtime_error("Unsupported data type");
                }
            }
    }
    else if((N == 2 && M == 8))
    {
        switch (input.type().scalarType()){
            case torch::ScalarType::Float: {
            const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
            prune_kernel<2, 8, float><<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size);
            break;
            }
            case torch::ScalarType::Half: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 8, at::Half><<<blocks, threads>>>(
                        input.data<at::Half>(),
                        output.data<at::Half>(),
                        mask.data<bool>(),
                        row_size);
                break;
            }
            case torch::ScalarType::BFloat16: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 8, at::BFloat16><<<blocks, threads>>>(
                        input.data<at::BFloat16>(),
                        output.data<at::BFloat16>(),
                        mask.data<bool>(),
                        row_size);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported data type");
            }
        }
    }
    else if((N == 2 && M == 16))
    {
        switch (input.type().scalarType()){
            case torch::ScalarType::Float: {
            const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
            prune_kernel<2, 16, float><<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size);
            break;
            }
            case torch::ScalarType::Half: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 16, at::Half><<<blocks, threads>>>(
                        input.data<at::Half>(),
                        output.data<at::Half>(),
                        mask.data<bool>(),
                        row_size);
            }
            case torch::ScalarType::BFloat16: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 16, at::BFloat16><<<blocks, threads>>>(
                        input.data<at::BFloat16>(),
                        output.data<at::BFloat16>(),
                        mask.data<bool>(),
                        row_size);
            }
            default: {
                throw std::runtime_error("Unsupported data type");
            }
        }
    }
    else
    {
        if(M < 8 || M % 8 != 0)
        {
            throw std::runtime_error("M must be a multiple of 8");
        }
        switch (input.type().scalarType()) {
            case torch::ScalarType::Float:
            {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size,
                    N,
                    M);
                 break;
            }
            case torch::ScalarType::Half:
            {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                    input.data<at::Half>(),
                    output.data<at::Half>(),
                    mask.data<bool>(),
                    row_size,
                    N,
                    M);
            }
        }
    }
  return {output, mask};
}


template <class T>
__global__ void prune_and_compress_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        int8_t* __restrict__ mask,
        size_t row_size) {
    const int input_column = 16 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int output_column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int input_row = blockIdx.y * row_size;
    const int output_row = blockIdx.y * (row_size / 2);
    const int input_index = input_row + input_column;
    const int output_index = output_row + output_column;
    if (input_column < row_size) {
        int8_t local_mask[16];
        reinterpret_cast<float4*>(local_mask)[0] = reinterpret_cast<const float4*>(&mask[input_index])[0];

        int local_index = 0;
        #pragma unroll (2)
        for(int i = 0; i < 2; i++)
        {
            T local_data[8];
            reinterpret_cast<float4*>(local_data)[0] = reinterpret_cast<const float4*>(&input[input_index + 8 * i])[0];
            #pragma unroll (8)
            for(int j = 0; j < 8; j++)
            {
                if(local_mask[8 * i + j] > 1)
                {
                    output[local_index + output_index] = local_data[j];
                    local_index++;
                }
            }
        }
    }
}


torch::Tensor prune_and_compress_cuda(torch::Tensor dense, torch::Tensor mask)
{
    auto row_size = dense.size(1);
    auto batch_size = dense.size(0);
    if(row_size % 16 != 0)
    {
        throw std::runtime_error("Pruning dimension should be a multiple of 128.");
    }
    auto options = torch::TensorOptions().dtype(dense.type().scalarType()).device(torch::kCUDA);
    torch::Tensor result = torch::zeros({dense.size(0), dense.size(1) / 2}, options);
    const int threads = 1024;
    switch (dense.type().scalarType()) {
        case torch::ScalarType::Float:
        {
            throw std::runtime_error("Full precision not supported for prune_and_compress");
        }
        case torch::ScalarType::Half:
        {
            const dim3 blocks(((row_size / 16) + threads - 1) / threads, batch_size);
            prune_and_compress_kernel<at::Half><<<blocks, threads>>>(
                dense.data<at::Half>(),
                result.data<at::Half>(),
                mask.data<int8_t>(),
                row_size);
            break;
        }
        case torch::ScalarType::BFloat16:
        {
            const dim3 blocks(((row_size / 16) + threads - 1) / threads, batch_size);
            prune_and_compress_kernel<at::BFloat16><<<blocks, threads>>>(
                dense.data<at::BFloat16>(),
                result.data<at::BFloat16>(),
                mask.data<int8_t>(),
                row_size);
            break;
        }
        default:
        {
            throw std::runtime_error("Unsupported data type");
        }
    }
    return result;
}