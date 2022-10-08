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


unsigned char *to_gray_scale(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}

void detect_cpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}

void detect_gpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}
