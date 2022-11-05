#include "detect_obj.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include "helpers_images.hpp"
#include "utils.hpp"

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
}

// (1 / 2*pi*sigma^2) / e(-(x^2 + y^2)/2 * sigma^2)
double **create_gaussian_kernel(unsigned char size) {
    double **kernel = create2Dmatrix<double>(size, size);
    int margin = (int) size / 2;
    double sigma = 1.0;

    double s = 2.0 * sigma * sigma;
    // sum is for normalization
    double sum = 0.0;

    for (int row = -margin; row <= margin; row++) {
        for (int col = -margin; col <= margin; col++) {
            const double radius = col * col + row * row;
            kernel[row + margin][col + margin] = (exp(-radius / s)) / (M_PI * s);
            sum += kernel[row + margin][col + margin];
        }
    }

    // normalising the Kernel
    for (unsigned char i = 0; i < size; ++i)
        for (unsigned char j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}

/*
unsigned char convolution(int x, int y, int height, int width, unsigned char **image, double** kernel, unsigned char kernel_size)
{
    double conv = 0;
    // get kernel val
    int margin = (int) kernel_size / 2;
    for(int i = -margin ; i <= margin; i++)
        for(int j = -margin; j <= margin; j++) {
            if (x + i < 0 || y + j < 0 || x + i >= height || y + j >= width)
                continue;
            conv += image[x + i][y + j] * kernel[i + margin][j + margin];
        }

    return conv;
}

unsigned char **apply_blurring(unsigned char** image, int width, int height, unsigned char kernel_size) {
    // Allocating for blurred image
    unsigned char** blurred_image = create2Dmatrix<unsigned char>(height, width);
    double **kernel = create_gaussian_kernel(kernel_size);
    
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            blurred_image[x][y] = convolution(x, y, height, width, image, kernel, kernel_size);
        }
   
    free2Dmatrix<double **>(kernel_size, kernel);
    return blurred_image; 
}
*/

__global__ void gaussian_blur(unsigned char *image, int rows, int cols, int kernel_size, double **kernel)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols)
        return;

   double conv = 0;
   // get kernel val
   int margin = (int) kernel_size / 2;
   for(int j = -margin ; j <= margin; j++)
       for(int k = -margin; k <= margin; k++) {
           if (i + k < 0 || i + j * cols < 0 || i + k >= cols || i + j * cols >= rows)
               continue;
           conv += image[i + k + (j * cols)] * kernel[j + margin][k + margin];
       }

   __syncthreads(); 

   image[i] = conv;
}

void to_save(unsigned char *buffer_cuda, int height, int width, std::string file) {
    size_t size = width * height * sizeof(unsigned char);
    unsigned char *b = (unsigned char *) std::malloc(size);

    cudaMemcpy(b, buffer_cuda, size, cudaMemcpyDeviceToHost);

    save(b, width, height, file);
    free(b);
}

unsigned char *malloc_cuda(size_t size) {
    unsigned char *buffer;
    cudaMalloc((void **) &buffer, size);
    cudaCheckError();

    return buffer;
}

unsigned char *cpy_host_to_device(unsigned char *buffer, size_t size) {
    unsigned char *device_buffer = malloc_cuda(size);

    cudaMemcpy(device_buffer, buffer, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    return device_buffer;
}

__global__ void to_gray_scale(unsigned char *buffer, unsigned char *result, int rows, int cols, int channels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols * channels)
        return;

    result[i] = (0.30 * buffer[i * channels] +         // R
                 0.59 * buffer[(i * channels) + 1] +   // G
                 0.11 * buffer[(i * channels) + 2]);   // B 
}

__global__ void difference(unsigned char *buffer_ref, unsigned char *buffer_obj, int rows, int cols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols)
        return;

    buffer_obj[i] = abs(buffer_ref[i] - buffer_obj[i]);
}

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels) {
    std::string file_save_gray_ref = "../images/gray_scale_ref_cuda.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj_cuda.jpg";

    std::string file_save_blur_ref = "../images/blurred_ref_cuda.jpg";
    std::string file_save_blur_obj = "../images/blurred_obj_cuda.jpg";

    std::string file_save_diff = "../images/diff_cuda.jpg";

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
    const int blocks_per_grid = (nb_of_elements + threads_per_bloks - 1) / threads_per_bloks;

    // Apply gray scale
    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(buffer_ref_cuda, gray_ref_cuda, rows, cols, channels);
    cudaCheckError();
    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(buffer_obj_cuda, gray_obj_cuda, rows, cols, channels);
    cudaCheckError();

    // Uncomment to see the images
    to_save(gray_ref_cuda, height, width, file_save_gray_ref);
    to_save(gray_obj_cuda, height, width, file_save_gray_obj);

    cudaFree(buffer_ref_cuda);
    cudaFree(buffer_obj_cuda);
    
    unsigned int kernel_size = 5;
    double **kernel = create_gaussian_kernel(kernel_size);

    gaussian_blur<<<blocks_per_grid, threads_per_bloks>>>(gray_ref_cuda, rows, cols, kernel_size, kernel);
    cudaCheckError();
    gaussian_blur<<<blocks_per_grid, threads_per_bloks>>>(gray_obj_cuda, rows, cols, kernel_size, kernel);
    cudaCheckError();

    to_save(gray_ref_cuda, height, width, file_save_blur_ref);
    to_save(gray_obj_cuda, height, width, file_save_blur_obj);
    // Difference
    difference<<<blocks_per_grid, threads_per_bloks>>>(gray_ref_cuda, gray_obj_cuda, rows, cols);
    
    // Uncomment to see the results
    to_save(gray_obj_cuda, height, width, file_save_diff);
}

