#pragma once
#include <iostream>

#include "utils_gpu.hpp"

__global__ void difference(unsigned char *buffer_ref, unsigned char *buffer_obj,
                           int rows, int cols, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int idx = y * pitch + x;
    buffer_obj[idx] = abs(buffer_ref[idx] - buffer_obj[idx]);
}
