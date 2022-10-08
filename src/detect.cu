#include "detect.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

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
unsigned char *to_gray_scale(unsigned char* buffer, int width, int height, int channels) {
    unsigned char *gray_scale = (unsigned char*)std::malloc(width * height * sizeof(unsigned char));

    int i = 0;
    int j = 0;
    for (;i < width * height * channels - channels; i += channels, j++) {
        gray_scale[j] = 0.3 * buffer[i] + 0.59 * buffer[i + 1] + 0.11 * buffer[i + 2];
    }

    return gray_scale;
}

void detect_cpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}

void detect_gpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}
