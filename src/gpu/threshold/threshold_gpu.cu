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

/* __global__ void apply_bin_threshold(unsigned char *buffer, unsigned char *buffer_bin, size_t rows, size_t cols, size_t pitch, int threshold) { */
/*     int col = blockDim.x * blockIdx.x + threadIdx.x; */
/*     int row = blockDim.y * blockIdx.y + threadIdx.y; */

/*     if (col >= cols || row >= rows) */
/*         return; */

/*     buffer_bin[col + row * pitch] = buffer[col + row * pitch]; */ 
/*     buffer[col + row * pitch] = 1 * (buffer[col + row * pitch] >= threshold); */ 
/* } */

/* __global__ void apply_bin_threshold2(unsigned int *buffer_bin, unsigned char *buffer_base, size_t rows, size_t cols, */
/*                                      size_t pitch, size_t pitch_bin, int threshold) { */
/*     unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; */
/*     unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; */

/*     if (col >= cols || row >= rows) */
/*         return; */

/*     unsigned int val = col + row * cols + 1; */
/*     unsigned int *b_bin = (unsigned int *)((char*)buffer_bin + row * pitch_bin + col * sizeof(unsigned int)); */

/*     if (buffer_base[col + row * pitch] >= threshold) */
/*         *b_bin = val; */
/*     else */
/*         *b_bin = 0; */
/* } */

template <typename T>
T* malloc2Dcuda(size_t rows, size_t cols, size_t *pitch) {
    T *buffer_device;
    cudaMallocPitch(&buffer_device, pitch, sizeof(T) * cols, rows);
    return buffer_device;
}

unsigned char threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    unsigned char otsu_thresh = otsu_threshold(buffer, rows, cols, pitch, thx, thy);
    /* unsigned char otsu_thresh2 = otsu_thresh * 2.5; */

    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    apply_first_threshold<<<blocks, threads>>>(buffer, rows, cols, pitch, otsu_thresh - 10);
    cudaDeviceSynchronize();
    cudaCheckError();

    return otsu_thresh < 102 ? otsu_thresh * 2.5 : 255;

    /* size_t pitch_bin; */
    /* unsigned int *buffer_bin = malloc2Dcuda<unsigned int>(rows, cols, &pitch_bin); */

    /* apply_bin_threshold2<<<blocks, threads>>>(buffer_bin, buffer, rows, cols, pitch, pitch_bin, otsu_thresh2); */
    /* cudaDeviceSynchronize(); */
    /* cudaCheckError(); */

    /* connexe_components(buffer, buffer_bin, rows, cols, pitch, pitch_bin, thx, thy); */
}

