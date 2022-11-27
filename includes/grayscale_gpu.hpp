#pragma once
#include <iostream>

#include "utils_gpu.hpp"

// buffer-1d, result-2d
/* __global__ void to_gray_scale(unsigned char *buffer, unsigned char *result, */
/*                               int rows, int cols, int channels, size_t pitch) */
/* { */
/*     int x = blockDim.x * blockIdx.x + threadIdx.x; */
/*     int y = blockDim.y * blockIdx.y + threadIdx.y; */

/*     if (x >= cols || y >= rows) */
/*         return; */

/*     int idx_chan = (y * cols + x) * channels; */

/*     result[y * pitch + x] = (0.30 * buffer[idx_chan] + // R */
/*                              0.59 * buffer[idx_chan + 1] + // G */
/*                              0.11 * buffer[idx_chan + 2]); // B */
/* } */
