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

__global__ void propagate(unsigned char *buffer_base, unsigned char *buffer_bin,
                          size_t rows, size_t cols, size_t pitch, bool *has_change) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    /* if (buffer_bin[col + row * pitch] == 0) */
    /*     return; */

    while (*has_change) {
        if (col == 0 && row == 0) {
            printf("HEE\n");
            *has_change = false;
        }

        __syncthreads();

        bool my_color = buffer_base[col + row * pitch] % 255 == 0;

        if (!my_color && (col + 1) < cols && (buffer_bin[col + 1 + row * pitch] == 255)) {
            buffer_bin[col + row * pitch] = 255;
            *has_change = true;
            printf("HHEEE\n");
        }
        /* if (!my_color && (col - 1) >= 0 && (buffer_bin[col - 1 + row * pitch] == 255)) { */
        /*     buffer_bin[col + row * pitch] = 255; */
        /*     *has_change = true; */
        /* } */
        /* if (!my_color && (row + 1) < rows && (buffer_bin[col + (row + 1) * pitch] == 255)) { */
        /*     buffer_bin[col + row * pitch] = 255; */
        /*     *has_change = true; */
        /* } */
        /* if (!my_color && (row - 1) >= 0 && (buffer_bin[col + (row - 1) * pitch] == 255)) { */
        /*     buffer_bin[col + row * pitch] = 255; */
        /*     *has_change = true; */
        /* } */

        __syncthreads();

        if (*has_change == true) {
            printf("TRUUUUE\n");
        }
    }
}

template <class T>
T *mallocCpy(T value) {
    T *device;
    cudaMalloc((void**) &device, sizeof(T));
    cudaMemcpy(device, &value, sizeof(T), cudaMemcpyHostToDevice);

    return device;
}

void connexe_components(unsigned char *buffer_base, unsigned char *buffer_bin,
                        size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    bool *has_change = mallocCpy<bool>(true);

   propagate<<<blocks, threads>>>(buffer_base, buffer_bin, rows, cols, pitch, has_change);
   cudaCheckError();
   cudaDeviceSynchronize();
}
