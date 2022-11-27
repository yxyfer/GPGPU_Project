#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"
#include <cassert>
#include <iostream>
#include <math.h>


float inf = std::numeric_limits<float>::infinity();

// See slides to improve
__global__ void sum_black_white(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold,
                                unsigned int *sum_white, unsigned int *sum_black) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    if (buffer[col + row * pitch] >= threshold) 
        atomicAdd(sum_white, buffer[col + row * pitch]);
    else
        atomicAdd(sum_black, buffer[col + row * pitch]);
    
}

__global__ void almost_var_black_white(unsigned char *buffer, size_t rows, size_t cols, size_t pitch,
                                       int threshold, float *mean_white, float *mean_black,
                                       float *sum_white, float *sum_black) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    if (buffer[col + row * pitch] >= threshold) {
        float val = ((float) buffer[col + row * pitch] - *mean_white) * ((float) buffer[col + row * pitch] - *mean_white);
        atomicAdd(sum_white, val);
    }
    else {
        float val = ((float) buffer[col + row * pitch] - *mean_black) * ((float) buffer[col + row * pitch] - *mean_black);
        atomicAdd(sum_black, val);
    }
}


__global__ void nb_bw_pixels(unsigned char *buffer, size_t rows, size_t cols, size_t pitch,
                          unsigned int *sum_white, unsigned int *sum_black, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows) {
        return;
    }

    if (buffer[col + row * pitch] >= threshold)
        atomicAdd(sum_white, 1);
    else
        atomicAdd(sum_black, 1);
}

__global__ void mean(float* dst, unsigned int* sum, unsigned int *size) {
    *dst = (float) *sum / (float) *size;
}

float otsu_criteria(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));
    
    unsigned int *d_nb_white_pi = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_nb_black_pi = mallocCpy<unsigned int>(0, sizeof(unsigned int));

    nb_bw_pixels<<<blocks, threads>>>(buffer, rows, cols, pitch, d_nb_white_pi, d_nb_black_pi, threshold);
    cudaCheckError();
    cudaDeviceSynchronize();
    
    unsigned int *d_sum_white = mallocCpy<unsigned int>(0, sizeof(unsigned int));
    unsigned int *d_sum_black = mallocCpy<unsigned int>(0, sizeof(unsigned int));

    sum_black_white<<<blocks, threads>>>(buffer, rows, cols, pitch, threshold, d_sum_white, d_sum_black);
    cudaCheckError();
    cudaDeviceSynchronize();

    float *d_mean_white = malloc1Dcuda<float>(sizeof(float));
    float *d_mean_black = malloc1Dcuda<float>(sizeof(float));
    
    mean<<<1, 1>>>(d_mean_white, d_sum_white, d_nb_white_pi);
    mean<<<1, 1>>>(d_mean_black, d_sum_black, d_nb_black_pi);
    cudaCheckError();
    cudaDeviceSynchronize();
    
    float *d_var_white = mallocCpy<float>(0, sizeof(float));
    float *d_var_black = mallocCpy<float>(0, sizeof(float));

    almost_var_black_white<<<blocks, threads>>>(buffer, rows, cols, pitch, threshold, d_mean_white, d_mean_black, d_var_white, d_var_black);
    cudaCheckError();
    cudaDeviceSynchronize();
   
    float h_var_white;
    float h_var_black;
    unsigned int h_nb_white_pi;
    unsigned int h_nb_black_pi;
    
    cudaMemcpy(&h_var_white, d_var_white, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_var_black, d_var_black, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nb_white_pi, d_nb_white_pi, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nb_black_pi, d_nb_black_pi, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    h_var_white /= h_nb_white_pi;
    h_var_black /= h_nb_black_pi;

    float h_weight_whitep = (float) h_nb_white_pi / (float) (rows * cols);
    float h_weight_blackp = 1.0 - (float) h_weight_whitep;

    cudaFree(d_nb_white_pi);
    cudaFree(d_nb_black_pi);
    cudaFree(d_sum_white);
    cudaFree(d_sum_black);
    cudaFree(d_var_white);
    cudaFree(d_var_black);

    if (h_weight_whitep == 0 || h_weight_blackp == 0)
        return inf;

    return(float) h_weight_whitep * h_var_white + (float) h_weight_blackp * h_var_black;
}

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    unsigned char opti_th = 0;
    float otsu_val = inf;

    for (unsigned char i = 0; i < 255; i++) {
        float otsu = otsu_criteria(buffer, rows, cols, pitch, i, thx, thy);
        if (otsu < otsu_val) {
            otsu_val = otsu;
            opti_th = i;
        }
    }

    return opti_th;
}
