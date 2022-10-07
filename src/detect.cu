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


void detect_cpu(char* buffer_start, char *buffer_obj, int width, int height, std::ptrdiff_t stride) {
    return;
}

void detect_gpu(char* buffer_start, char *buffer_obj, int width, int height, std::ptrdiff_t stride) {
    return;
}
