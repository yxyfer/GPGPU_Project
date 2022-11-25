#pragma once

#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>

#include "utils_gpu.hpp"

unsigned char *circular_kernel_gpu(int kernel_size)
{
    unsigned char *kernel = (unsigned char *)malloc(kernel_size * kernel_size
                                                    * sizeof(unsigned char));
    memset(kernel, 0, kernel_size * kernel_size);
    int radius = kernel_size / 2;
    for (int x = -radius; x < radius + 1; x++)
    {
        int y = (std::sqrt(radius * radius - (x * x)));
        for (int j = -y; j < y + 1; j++)
        {
            kernel[(j + radius) * kernel_size + (x + radius)] = 1;
            kernel[(-j + radius) * kernel_size + (x + radius)] = 1;
        }
    }
    size_t kernel_gpu_size = kernel_size * kernel_size * sizeof(unsigned char);
    unsigned char *kernel_gpu =
        cpy_host_to_device<unsigned char>(kernel, kernel_gpu_size);
    return kernel_gpu;
}

__global__ void perform_dilation_gpu(unsigned char *image, int rows, int cols,
                                     size_t kernel_size, unsigned char *kernel,
                                     int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int start_k = kernel_size / 2;

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)]
                    * kernel[(j + start_k) * kernel_size + (i + start_k)];
                if (mult > res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

__global__ void perform_erosion_gpu(unsigned char *image, int rows, int cols,
                                    size_t kernel_size, unsigned char *kernel,
                                    int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int start_k = kernel_size / 2;

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)]
                    * kernel[(j + start_k) * kernel_size + (i + start_k)];
                if (mult != 0 && (res == 0 || mult < res))
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

/*

__global__ void perform_dilation_gpu(unsigned char *image, int rows, int cols,
                                     size_t kernel_size, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int radius = kernel_size / 2;
    int res = 0;

    for (int i = -radius; i < radius + 1; i++)
    {
        int k_y = sqrt(float(radius * radius - (i * i)));
        for (int j = -k_y; j < k_y + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)];
                if (mult > res)
                    res = mult;

                mult = image[(y - j) * pitch + (x + i)];
                if (mult > res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

__global__ void perform_erosion_gpu(unsigned char *image, int rows, int cols,
                                    size_t kernel_size, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int radius = kernel_size / 2;
    int res = 0;

    for (int i = -radius; i < radius + 1; i++)
    {
        int k_y = sqrt(float(radius * radius - (i * i)));
        for (int j = -k_y; j < k_y + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)];
                if (res == 0 || mult < res)
                    res = mult;

                mult = image[(y - j) * pitch + (x + i)];
                if (res == 0 || mult < res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}
*/
