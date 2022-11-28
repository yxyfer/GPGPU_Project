#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"

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

void difference_gpu(unsigned char *ref, unsigned char *obj, size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    const dim3 threads(thx, thy);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)), std::ceil(float(rows) / float(threads.y)));
    
    difference<<<blocks, threads>>>(ref, obj, rows, cols, pitch);

    cudaCheckError();
    cudaDeviceSynchronize();
}
