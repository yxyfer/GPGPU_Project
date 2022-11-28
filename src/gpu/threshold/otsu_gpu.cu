#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"
#include <cassert>
#include <iostream>
#include <math.h>


float inf = std::numeric_limits<float>::infinity();

// See slides to improve
__global__ void otsu(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold,
                     unsigned int *nb_white, unsigned int *nb_black,
                     unsigned int *sum_white, unsigned int *sum_black,
                     unsigned int *sum2_white, unsigned int *sum2_black) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows) {
        return;
    }
    
    __shared__ unsigned int local_nb_white;
    __shared__ unsigned int local_nb_black;
    
    __shared__ unsigned int local_sum_white;
    __shared__ unsigned int local_sum_black;
    
    __shared__ unsigned int local_sum2_white;
    __shared__ unsigned int local_sum2_black;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_nb_white = 0;
        local_nb_black = 0;
        local_sum_black = 0;
        local_sum_white = 0;
        local_sum2_black = 0;
        local_sum2_white = 0;
    }

    __syncthreads();

    unsigned char val = buffer[col + row * pitch];

    if (val >= threshold) {
        atomicAdd(&local_nb_white, 1);
        atomicAdd(&local_sum_white, val);
        atomicAdd(&local_sum2_white, val * val);
    }
    else {
        atomicAdd(&local_nb_black, 1);
        atomicAdd(&local_sum_black, val);
        atomicAdd(&local_sum2_black, val * val);
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(nb_white, local_nb_white);
        atomicAdd(nb_black, local_nb_black);
        atomicAdd(sum_white, local_sum_white);
        atomicAdd(sum_black, local_sum_black);
        atomicAdd(sum2_black, local_sum2_black);
        atomicAdd(sum2_white, local_sum2_white);
    }
}

__global__ void final_work(unsigned int *nb_white, unsigned int *nb_black,
                      unsigned int *sum_white, unsigned int *sum_black,
                      unsigned int *sum2_white, unsigned int *sum2_black,
                      float *result) {

    float mean_white = (float) *sum_white / (float) *nb_white;
    float mean_black = (float) *sum_black / (float) *nb_black;

    float var_white = (float) *sum2_white / (float) *nb_white - (mean_white * mean_white);
    float var_black = (float) *sum2_black / (float) *nb_black - (mean_black * mean_black);

    float weight_whitep = (float) *nb_white / (float) (*nb_black + *nb_white);
    float weight_blackp = 1.0 - weight_whitep;

    if (weight_blackp != 0 && weight_whitep != 0)
        *result = weight_whitep * var_white + weight_blackp * var_black;
}

float otsu_criteria(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));
    
    unsigned int *d_nb_white_pi = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_nb_black_pi = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_sum_white = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_sum_black = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_sum2_white = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_sum2_black = mallocCpy<unsigned int>(0, sizeof(unsigned int));

    float *d_result = mallocCpy<float>(inf, sizeof(float));

    otsu<<<blocks, threads>>>(buffer, rows, cols, pitch, threshold,
                              d_nb_white_pi, d_nb_black_pi,
                              d_sum_white, d_sum_black,
                              d_sum2_white, d_sum2_black);
    /* cudaCheckError(); */
    /* cudaDeviceSynchronize(); */

    final_work<<<1, 1>>>(d_nb_white_pi, d_nb_black_pi,
                         d_sum_white, d_sum_black,
                         d_sum2_white, d_sum2_black,
                         d_result);
    
    /* cudaCheckError(); */
    /* cudaDeviceSynchronize(); */


    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    return h_result;
}

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    unsigned char opti_th = 0;
    float otsu_val = inf;

    for (unsigned char i = 10; i < 103; i++) {
        float otsu = otsu_criteria(buffer, rows, cols, pitch, i, thx, thy);
        if (otsu < otsu_val) {
            otsu_val = otsu;
            opti_th = i;
        }
    }
    
    cudaCheckError();
    cudaDeviceSynchronize();

    return opti_th;
}
