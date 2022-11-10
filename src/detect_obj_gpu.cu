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

void to_save(unsigned char *buffer_cuda, int rows, int cols, std::string file,
             size_t pitch)
{
    size_t size = rows * cols * sizeof(unsigned char);
    unsigned char *host_buffer = (unsigned char *)std::malloc(size);

    cudaMemcpy2D(host_buffer, cols * sizeof(unsigned char), buffer_cuda, pitch,
                 cols * sizeof(unsigned char), rows, cudaMemcpyDeviceToHost);

    save(host_buffer, cols, rows, file);
    free(host_buffer);
}

unsigned char *malloc_cuda(size_t size)
{
    unsigned char *buffer;
    cudaMalloc((void **)&buffer, size);
    cudaCheckError();

    return buffer;
}

unsigned char *malloc_cuda2D(size_t cols, size_t rows, size_t *pitch)
{
    unsigned char *buffer;
    // Allocate an 2D buffer with padding
    //@@ use cudaMallocPitch to allocate this buffer
    cudaMallocPitch(&buffer, pitch, cols * sizeof(unsigned char), rows);
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

// buffer-1d, result-2d
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

int get_properties(int device, cudaDeviceProp *deviceProp)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (device >= deviceCount)
        return -1;

    cudaGetDeviceProperties(deviceProp, device);
    if (deviceProp->major == 9999 && deviceProp->minor == 9999)
    {
        std::cerr << "No CUDA GPU has been detected" << std::endl;
        return -1;
    }
    return 0;
}

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels)
{
    std::string file_save_gray_ref = "../images/gray_scale_ref_cuda.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj_cuda.jpg";
    std::string file_save_diff = "../images/diff_cuda.jpg";

    const int rows = height;
    const int cols = width;

    cudaDeviceProp deviceProp;
    int gpu_error = get_properties(0, &deviceProp);

    const dim3 threadsPerBlock = dim3(std::sqrt(deviceProp.maxThreadsDim[0]),
                                      std::sqrt(deviceProp.maxThreadsDim[1]));
    const dim3 blocksPerGrid =
        dim3(std::ceil(float(cols) / float(threadsPerBlock.x)),
             std::ceil(float(rows) / float(threadsPerBlock.y)));

    std::cout << "Threads per block: (" << threadsPerBlock.x << ", "
              << threadsPerBlock.y << ")\n";
    std::cout << "Blocks per grid: (" << blocksPerGrid.x << ", "
              << blocksPerGrid.y << ")\n";

    const size_t size_color = cols * rows * channels * sizeof(unsigned char);

    // Copy content of host buffer to a cuda buffer
    unsigned char *buffer_ref_cuda = cpy_host_to_device(buffer_ref, size_color);
    unsigned char *buffer_obj_cuda = cpy_host_to_device(buffer_obj, size_color);

    size_t pitch; // we will store the pitch value in this variable
    unsigned char *gray_ref_cuda = malloc_cuda2D(cols, rows, &pitch);
    unsigned char *gray_obj_cuda = malloc_cuda2D(cols, rows, &pitch);

    std::cout << "Pitch: " << pitch << '\n';

    // Apply gray scale
    to_gray_scale<<<blocksPerGrid, threadsPerBlock>>>(
        buffer_ref_cuda, gray_ref_cuda, rows, cols, channels, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();

    to_gray_scale<<<blocksPerGrid, threadsPerBlock>>>(
        buffer_obj_cuda, gray_obj_cuda, rows, cols, channels, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();

    // Uncomment to see the images
    to_save(gray_ref_cuda, rows, cols, file_save_gray_ref, pitch);
    to_save(gray_obj_cuda, height, width, file_save_gray_obj);

    cudaFree(buffer_ref_cuda);
    cudaFree(buffer_obj_cuda);

    // Difference
    // difference<<<blocksPerGrid, threadsPerBlock>>>(gray_ref_cuda,
    // gray_obj_cuda,
    //                                               rows, cols, pitch);

    // Uncomment to see the results
    // to_save(gray_obj_cuda, height, width, file_save_diff, pitch);
}
