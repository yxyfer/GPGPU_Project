#include <cassert>
#include <cmath>
#include <iostream>

#include "detect_obj.hpp"
#include "helpers_images.hpp"
#include "utils.hpp"

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

// (1 / 2*pi*sigma^2) / e(-(x^2 + y^2)/2 * sigma^2)
double **create_gaussian_kernel(unsigned char size)
{
    double **kernel = create2Dmatrix<double>(size, size);
    int margin = (int)size / 2;
    double sigma = 1.0;

    double s = 2.0 * sigma * sigma;
    // sum is for normalization
    double sum = 0.0;

    for (int row = -margin; row <= margin; row++)
    {
        for (int col = -margin; col <= margin; col++)
        {
            const double radius = col * col + row * row;
            kernel[row + margin][col + margin] =
                (exp(-radius / s)) / (M_PI * s);
            sum += kernel[row + margin][col + margin];
        }
    }

    // normalising the Kernel
    for (unsigned char i = 0; i < size; ++i)
        for (unsigned char j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}

__global__ void gaussian_blur(unsigned char *image, int rows, int cols,
                              int kernel_size, double **kernel)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols)
        return;

    double conv = 0;
    // get kernel val
    int margin = (int)kernel_size / 2;
    for (int j = -margin; j <= margin; j++)
        for (int k = -margin; k <= margin; k++)
        {
            if (i + k < 0 || i + j * cols < 0 || i + k >= cols
                || i + j * cols >= rows)
                continue;
            conv += image[i + k + (j * cols)] * kernel[j + margin][k + margin];
        }

    __syncthreads();

    image[i] = conv;
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

    std::string file_save_blur_ref = "../images/blurred_ref_cuda.jpg";
    std::string file_save_blur_obj = "../images/blurred_obj_cuda.jpg";

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

    unsigned int kernel_size = 5;
    double **kernel = create_gaussian_kernel(kernel_size);

    gaussian_blur<<<blocks_per_grid, threads_per_bloks>>>(
        gray_ref_cuda, rows, cols, kernel_size, kernel);
    cudaCheckError();
    gaussian_blur<<<blocks_per_grid, threads_per_bloks>>>(
        gray_obj_cuda, rows, cols, kernel_size, kernel);
    cudaCheckError();

    to_save(gray_ref_cuda, height, width, file_save_blur_ref);
    to_save(gray_obj_cuda, height, width, file_save_blur_obj);
    // Difference
    // difference<<<blocksPerGrid, threadsPerBlock>>>(gray_ref_cuda,
    // gray_obj_cuda,
    //                                               rows, cols, pitch);

    // Uncomment to see the results
    // to_save(gray_obj_cuda, height, width, file_save_diff, pitch);
}
