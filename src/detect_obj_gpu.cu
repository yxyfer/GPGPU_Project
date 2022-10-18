#include "detect_obj.hpp"
#include <cassert>
#include <iostream>
#include "helpers_images.hpp"

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
}

void to_save(unsigned char *buffer_cuda, int height, int width, std::string file) {
    size_t size = width * height * sizeof(unsigned char);
    unsigned char *b = (unsigned char *) std::malloc(size);

    cudaMemcpy(b, buffer_cuda, size, cudaMemcpyDeviceToHost);

    save(b, width, height, file);
    free(b);
}

__global__ void to_gray_scale(unsigned char *buffer, unsigned char *result, int rows, int cols, int channels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows * cols * channels)
        return;

    result[i] = (0.30 * buffer[i * channels] + 0.59 * buffer[(i * channels) + 1] + 0.11 * buffer[(i * channels) + 2]);   // RGB 
}

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels) {
    const int rows = height;
    const int cols = width;
    size_t size_color = cols * rows * channels * sizeof(unsigned char);
    unsigned char *buffer_ref_cuda;
    unsigned char *buffer_obj_cuda;

    cudaMalloc((void **) &buffer_ref_cuda, size_color);
    cudaCheckError();
    cudaMalloc((void **) &buffer_obj_cuda, size_color);
    cudaCheckError();

    cudaMemcpy(buffer_ref_cuda, buffer_ref, size_color, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(buffer_obj_cuda, buffer_obj, size_color, cudaMemcpyHostToDevice);
    cudaCheckError();
   
    size_t nb_of_elements = cols * rows;
    size_t size = nb_of_elements * sizeof(unsigned char);
    unsigned char *gray_ref_cuda;
    unsigned char *gray_obj_cuda;
    
    cudaMalloc((void **) &gray_ref_cuda, size);
    cudaCheckError();
    cudaMalloc((void **) &gray_obj_cuda, size);
    cudaCheckError();
    
    int threads_per_bloks = 1024;
    int blocks_per_grid = (nb_of_elements + threads_per_bloks - 1) / threads_per_bloks;

    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(buffer_ref_cuda, gray_ref_cuda, rows, cols, channels);
    cudaCheckError();
    to_gray_scale<<<blocks_per_grid, threads_per_bloks>>>(buffer_obj_cuda, gray_obj_cuda, rows, cols, channels);
    cudaCheckError();
    
    std::string file_save_gray_ref = "../images/gray_scale_ref_cuda.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj_cuda.jpg";

    to_save(gray_ref_cuda, height, width, file_save_gray_ref);
    to_save(gray_obj_cuda, height, width, file_save_gray_obj);
}

