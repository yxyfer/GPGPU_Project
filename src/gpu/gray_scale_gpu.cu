#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"

__global__ void to_gray_scale(unsigned char *buffer, unsigned char *result,
                              int rows, int cols, int channels, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int idx_chan = (y * cols + x) * channels;

    result[y * pitch + x] = (0.30 * buffer[idx_chan] + // R
                             0.59 * buffer[idx_chan + 1] + // G
                             0.11 * buffer[idx_chan + 2]); // B
}

void gray_scale_gpu(unsigned char *buffer_cuda, unsigned char *gray_cuda,
                    int rows, int cols, int pitch, int channels, int thx, int thy) {

    const dim3 threads(thx, thy);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)), std::ceil(float(rows) / float(threads.y)));

    to_gray_scale<<<blocks, threads>>>(buffer_cuda, gray_cuda, rows, cols, channels, pitch);

    cudaDeviceSynchronize();
    cudaCheckError();
}

void gray_scale_test(unsigned char *buffer, int width, int height, int channels) {
    const int rows = height;
    const int cols = width;

    const dim3 threads(32, 32);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)), std::ceil(float(rows) / float(threads.y)));

    const size_t size_color = cols * rows * channels * sizeof(unsigned char);

    unsigned char *buffer_ref_cuda = cpyHostToDevice<unsigned char>(buffer, size_color);

    size_t pitch; // we will store the pitch value in this variable
    unsigned char *gray_cuda = malloc2Dcuda<unsigned char>(cols, rows, &pitch);

    gray_scale_gpu(buffer_ref_cuda, gray_cuda, rows, cols, pitch, channels, threads.x, threads.y);

    cudaFree(buffer_ref_cuda);
    cudaFree(gray_cuda);
}

