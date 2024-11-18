#include <cuda.h>
 #include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void calculate_half_min_max(at::Half* data_pointer, at::Half* mins_data, at::Half* maxs_data, int num_cols){
   __half min = data_pointer[0];
   __half max = data_pointer[0];
    for (int i = 0; i < num_cols/(8 * 32); i++){
        float4 curr_data = reinterpret_cast<float4*>(data_pointer + (blockIdx.x * num_cols)+ (i*256) + (threadIdx.x *8) )[0];
        __half * half_data = reinterpret_cast<__half*>(&curr_data);

//         min =static_cast<__half>(half_data[0]);
//         max =static_cast<__half>( half_data[0]);

        for (int j = 0; j < 8; j++){
            min = __hmin(min, half_data[j]);
            max = __hmax(max, half_data[j]);
        }
        for (int s = 1; s < 32; s*=2){
            min = __hmin(min, __shfl_down_sync(0xFFFFFFFF, min, s, 32));
            max = __hmax(max, __shfl_down_sync(0xFFFFFFFF, max, s, 32));
        }
    }
    if (threadIdx.x == 0){
//         printf("%f \n", __half2float(min));
//         printf("%f \n", __half2float(max));
        maxs_data[blockIdx.x] = static_cast<at::Half>(max);
        mins_data[blockIdx.x] = static_cast<at::Half>(min);
    }

}

void get_half_min_max_cuda(torch::Tensor mat, torch::Tensor mins, torch::Tensor maxs){

    at::Half* data_pointer = mat.data_ptr<at::Half>();
    at::Half* maxs_data = maxs.data_ptr<at::Half>();
    at::Half* mins_data = mins.data_ptr<at::Half>();

    int num_rows = mat.sizes()[0];
    int num_cols = mat.sizes()[1];
    int num_blocks = num_rows;
    int num_threads = 32;

    calculate_half_min_max<<<num_blocks, num_threads>>>(data_pointer, mins_data, maxs_data, num_cols);
}

__global__ void calculate_float_min_max(float* data_pointer, float* mins_data, float* maxs_data, int num_cols){
    float min = data_pointer[0];
    float max = data_pointer[0];
    for (int i = 0; i < num_cols/(4 * 32); i++){
        float4 curr_data = reinterpret_cast<float4*>(data_pointer + (blockIdx.x * num_cols)+ (i*128) + (threadIdx.x *4) )[0];
        float * float_data = reinterpret_cast<float*>(&curr_data);

        for (int j = 0; j < 4; j++){
            min = fmin(min, float_data[j]);
            max = fmax(max, float_data[j]);
        }
        for (int s = 1; s < 32; s*=2){
            min = fmin(min, __shfl_down_sync(0xFFFFFFFF, min, s, 32));
            max = fmax(max, __shfl_down_sync(0xFFFFFFFF, max, s, 32));
        }
    }
    if (threadIdx.x == 0){
        maxs_data[blockIdx.x] = static_cast<float>(max);
        mins_data[blockIdx.x] = static_cast<float>(min);
    }

}

void get_float_min_max_cuda(torch::Tensor mat, torch::Tensor mins, torch::Tensor maxs){

    float * data_pointer = mat.data_ptr<float>();
    float * maxs_data = maxs.data_ptr<float>();
    float * mins_data = mins.data_ptr<float>();

    int num_rows = mat.sizes()[0];
    int num_cols = mat.sizes()[1];
    int num_blocks = num_rows;
    int num_threads = 32;

    calculate_float_min_max<<<num_blocks, num_threads>>>(data_pointer, mins_data, maxs_data, num_cols);
}