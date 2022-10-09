#include "detect.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


// Luminosity Method: gray scale -> 0.3 * R + 0.59 * G + 0.11 * B;
unsigned char **to_gray_scale(unsigned char* buffer, int width, int height, int channels) {
    unsigned char **gray_scale = (unsigned char **) malloc(height * sizeof(unsigned char *));
    for(int i = 0; i < height; i++)
        gray_scale[i] = (unsigned char *) malloc(width * sizeof(unsigned char));
    
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            gray_scale[r][c] = (0.30 * buffer[(r * width + c) * 3] +       // R
                                0.59 * buffer[(r * width + c) * 3 + 1] +   // G
                                0.11 * buffer[(r * width + c) * 3 + 2]);   // B
        }
    }

    return gray_scale;
}

void detect_cpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}

void detect_gpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}
