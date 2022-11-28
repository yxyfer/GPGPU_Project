#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"
#include <cassert>
#include <iostream>

__global__ void apply_first_threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows || buffer[col + row * pitch] >= threshold)
        return;

    buffer[col + row * pitch] = 0; 
}

unsigned char threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    unsigned char otsu_thresh = otsu_threshold(buffer, rows, cols, pitch, thx, thy);

    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    apply_first_threshold<<<blocks, threads>>>(buffer, rows, cols, pitch, otsu_thresh - 10);
    cudaDeviceSynchronize();
    cudaCheckError();

    return otsu_thresh < 102 ? otsu_thresh * 2.5 : 255;
}

