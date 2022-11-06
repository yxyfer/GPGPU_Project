#include <cstdio>
#include <numeric>

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

#include <cmath>
#include <iostream>

#include "detect_obj.hpp"

/*
Verify if the kernel is included in the image at the position (i, j)
*/

__global__ void perform_dilation_gpu(char *in, char *out, char *kernel,
                                     int height, int width, int k_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_k = k_size / 2;

    int res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if (y + j >= 0 && y + j < height && x + i >= 0 && x + i < width)
            {
                int mult = out[(y + j) * width + (x + i)]
                    * kernel[(i + start_k) * k_size + (j + start_k)];
                if (mult > res)
                    res = mult;
            }
        }
    }

    out[y * width + x] = res;
}

__global__ void perform_erosion_gpu(char *in, char *out, char *kernel,
                                    int height, int width, int k_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_k = k_size / 2;

    int res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if (y + j >= 0 && y + j < height && x + i >= 0 && x + i < width)
            {
                int mult = out[(y + j) * width + (x + i)]
                    * kernel[(i + start_k) * k_size + (j + start_k)];
                if (res == 0 || mult < res)
                    res = mult;
            }
        }
    }

    out[y * width + x] = res;
}
