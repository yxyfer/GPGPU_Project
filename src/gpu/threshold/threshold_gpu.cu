#include "threshold_gpu.hpp"

#include <cassert>
#include <iostream>

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

__global__ void apply_first_threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows || buffer[col + row * pitch] >= threshold)
        return;

    buffer[col + row * pitch] = 0; 
}

__global__ void apply_bin_threshold(unsigned char *buffer, unsigned char *buffer_bin, size_t rows, size_t cols, size_t pitch, int threshold) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    buffer_bin[col + row * pitch] = buffer[col + row * pitch]; 
    buffer[col + row * pitch] = 255 * (buffer[col + row * pitch] >= threshold); 
}

unsigned char* malloc2Dcuda(size_t rows, size_t cols, size_t *pitch) {
    unsigned char *buffer_device;
    cudaMallocPitch(&buffer_device, pitch, sizeof(unsigned char) * cols, rows);
    return buffer_device;
}

void threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    std::cout << "Start threshold\n";
    unsigned char otsu_thresh = otsu_threshold(buffer, rows, cols, pitch, thx, thy);
    unsigned char otsu_thresh2 = otsu_thresh * 2.5;

    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    apply_first_threshold<<<blocks, threads>>>(buffer, rows, cols, pitch, otsu_thresh - 10);
    cudaCheckError();
    cudaDeviceSynchronize();

    size_t pitch2;
    unsigned char *buffer_base = malloc2Dcuda(rows, cols, &pitch2);

    apply_bin_threshold<<<blocks, threads>>>(buffer, buffer_base, rows, cols, pitch, otsu_thresh2);
    cudaCheckError();
    cudaDeviceSynchronize();

    connexe_components(buffer_base, buffer, rows, cols, pitch, thx, thy);
    cudaFree(buffer_base);
    std::cout << "End threshold\n";
}

