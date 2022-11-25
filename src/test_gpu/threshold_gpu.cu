#include "gpu_functions.hpp"
#include <cassert>
#include <iostream>

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

int threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch) {
    unsigned char otsu_thresh = otsu_threshold(base_image);
    unsigned char otsu_thresh2 = otsu_thresh * 2.5;

    dim3 threads(32,32);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1)/ threads.y);

    apply_first_threshold<<<blocks, threads>>>(buffer, rows, cols, pitch, otsu_thresh - 10);

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

