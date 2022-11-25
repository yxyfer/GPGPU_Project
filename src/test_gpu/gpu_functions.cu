#include "gpu_functions.hpp"
#include <cassert>
#include <iostream>
#include <math.h>

float inf = std::numeric_limits<float>::infinity();

// See slides to improve
__global__ void sum(unsigned char *buffer, size_t size, int *sum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;

    atomicAdd(sum, buffer[x]);
}

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

// See slides to improve
__global__ void almost_var(unsigned char *buffer, size_t size, int *sum, int mean) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;

    int val = pow(buffer[x] - mean, 2);
    atomicAdd(sum, val);
}

__global__ void apply_first_threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows || buffer[col + row * pitch] >= threshold)
        return;

    buffer[col + row * pitch] = 0; 
}

__global__ void apply_bin_threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    buffer[col + row * pitch] = 255 * (buffer[col + row * pitch] >= threshold); 
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

template <class T>
T *mallocCpy(T value) {
    T *device;
    cudaMalloc((void**) &device, sizeof(T));
    cudaMemcpy(device, &value, sizeof(T), cudaMemcpyHostToDevice);

    return device;
}

template <class T>
T *mallocSimple() {
    T *device;
    cudaMalloc((void**) &device, sizeof(T));

    return device;
}

float otsu_criteria(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold) {
    dim3 threads(32,32);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1)/ threads.y);
    
    unsigned int *d_nb_white_pi = mallocCpy<unsigned int>(0);
    unsigned int *d_nb_black_pi = mallocCpy<unsigned int>(0);

    nb_bw_pixels<<<blocks, threads>>>(buffer, rows, cols, pitch, d_nb_white_pi, d_nb_black_pi, threshold);
    cudaDeviceSynchronize();
    
    unsigned int *d_sum_white = mallocCpy<unsigned int>(0);
    unsigned int *d_sum_black = mallocCpy<unsigned int>(0);

    sum_black_white<<<blocks, threads>>>(buffer, rows, cols, pitch, threshold, d_sum_white, d_sum_black);
    cudaDeviceSynchronize();

    float *d_mean_white = mallocSimple<float>();
    float *d_mean_black = mallocSimple<float>();
    
    mean<<<1, 1>>>(d_mean_white, d_sum_white, d_nb_white_pi);
    mean<<<1, 1>>>(d_mean_black, d_sum_black, d_nb_black_pi);
    cudaDeviceSynchronize();
    
    float *d_var_white = mallocCpy<float>(0);
    float *d_var_black = mallocCpy<float>(0);

    almost_var_black_white<<<blocks, threads>>>(buffer, rows, cols, pitch, threshold, d_mean_white, d_mean_black, d_var_white, d_var_black);
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

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch) {
    unsigned char opti_th = 0;
    float otsu_val = inf;

    for (unsigned char i = 0; i < 255; i++) {
        float otsu = otsu_criteria(buffer, rows, cols, pitch, i);
        /* std::cout << i << " | " << otsu << '\n'; */
        if (otsu < otsu_val) {
            otsu_val = otsu;
            opti_th = i;
        }
    }

    return opti_th;
}

unsigned char* malloc2Dcuda(size_t rows, size_t cols, size_t *pitch) {
    unsigned char *buffer_device;
    cudaMallocPitch(&buffer_device, pitch, sizeof(unsigned char) * cols, rows);
    return buffer_device;
}

unsigned char* malloc1Dcuda(size_t size) {
    unsigned char *buffer;
    cudaMalloc((void **) &buffer, size * sizeof(unsigned char));
    return buffer;
}

void cpyHostToGpu2D(unsigned char* buffer_host, unsigned char* buffer_gpu, size_t rows, size_t cols, size_t pitch) {
    /* cudaMemcpy2D(buffer_gpu, cols * sizeof(unsigned char), */
    /*         buffer_host, pitch, cols * sizeof(unsigned char), rows, cudaMemcpyHostToDevice); */
    cudaMemcpy2D(buffer_gpu, pitch, buffer_host, cols * sizeof(unsigned char), cols * sizeof(unsigned char), rows, cudaMemcpyHostToDevice);
}

int test_function(unsigned char *buffer, size_t rows, size_t cols) {
    size_t pitch;

    unsigned char *buffer_gpu = malloc2Dcuda(rows, cols, &pitch);
    cpyHostToGpu2D(buffer, buffer_gpu, rows, cols, pitch);

    /* dim3 threads(32,32); */
    /* dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1)/ threads.y); */

    /* apply_first_threshold<<<blocks, threads>>>(buffer_gpu, rows, cols, pitch, 20); */
    /* apply_bin_threshold<<<blocks, threads>>>(buffer_gpu, rows, cols, pitch, 20); */

    int threshold = otsu_threshold(buffer_gpu, rows, cols, pitch);
    /* int threshold = otsu_criteria(buffer_gpu, rows, cols, pitch, 10); */

    std::cout << threshold << '\n';

    return 0;
}

