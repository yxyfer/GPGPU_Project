#include <cassert>
#include <iostream>

#include "detect_obj.hpp"
#include "helpers_images.hpp"

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

void to_save(unsigned char *buffer_cuda, int height, int width,
             std::string file)
{
    size_t size = width * height * sizeof(unsigned char);
    unsigned char *b = (unsigned char *)std::malloc(size);

    cudaMemcpy(b, buffer_cuda, size, cudaMemcpyDeviceToHost);

    save(b, width, height, file);
    free(b);
}

unsigned char *malloc_cuda(size_t size)
{
    unsigned char *buffer;
    cudaMalloc((void **)&buffer, size);
    cudaCheckError();

    return buffer;
}

unsigned char *cpy_host_to_device(unsigned char *buffer, size_t size)
{
    unsigned char *device_buffer = malloc_cuda(size);

    cudaMemcpy(device_buffer, buffer, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    return device_buffer;
}

__global__ void to_gray_scale(unsigned char *buffer, unsigned char *result,
                              int rows, int cols, int channels)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols * channels)
        return;

    result[i] = (0.30 * buffer[i * channels] + // R
                 0.59 * buffer[(i * channels) + 1] + // G
                 0.11 * buffer[(i * channels) + 2]); // B
}

__global__ void difference(unsigned char *buffer_ref, unsigned char *buffer_obj,
                           int rows, int cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols)
        return;

    buffer_obj[i] = abs(buffer_ref[i] - buffer_obj[i]);
}

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels)
{
    /* std::string file_save_gray_ref = "../images/gray_scale_ref_cuda.jpg"; */
    /* std::string file_save_gray_obj = "../images/gray_scale_obj_cuda.jpg"; */
    /* std::string file_save_diff = "../images/diff_cuda.jpg"; */

    const int rows = height;
    const int cols = width;

    const size_t size_color = cols * rows * channels * sizeof(unsigned char);

    // Copy content of host buffer to a cuda buffer
    unsigned char *buffer_ref_cuda = cpy_host_to_device(buffer_ref, size_color);
    unsigned char *buffer_obj_cuda = cpy_host_to_device(buffer_obj, size_color);

    size_t nb_of_elements = cols * rows;
    size_t size = nb_of_elements * sizeof(unsigned char);
    unsigned char *gray_ref_cuda = malloc_cuda(size);
    unsigned char *gray_obj_cuda = malloc_cuda(size);

    const int threads_per_bloks = 1024;
    const int blocks_per_grid =
        (nb_of_elements + threads_per_bloks - 1) / threads_per_bloks;

    // Apply gray scale
    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(
        buffer_ref_cuda, gray_ref_cuda, rows, cols, channels);
    cudaCheckError();
    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(
        buffer_obj_cuda, gray_obj_cuda, rows, cols, channels);
    cudaCheckError();

    // Uncomment to see the images
    /* to_save(gray_ref_cuda, height, width, file_save_gray_ref); */
    /* to_save(gray_obj_cuda, height, width, file_save_gray_obj); */

    cudaFree(buffer_ref_cuda);
    cudaFree(buffer_obj_cuda);

    // Difference
    difference<<<blocks_per_grid, threads_per_bloks>>>(
        gray_ref_cuda, gray_obj_cuda, rows, cols);

    // Uncomment to see the results
    /* to_save(gray_obj_cuda, height, width, file_save_diff); */
}
