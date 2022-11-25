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
    apply_bin_threshold<<<blocks, threads>>>(buffer, rows, cols, pitch, otsu_thresh2) {
}

