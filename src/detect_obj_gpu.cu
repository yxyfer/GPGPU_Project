#include <cassert>
#include <cmath>
#include <iostream>

#include "blur_gpu.hpp"
#include "detect_obj_gpu.hpp"
#include "difference_gpu.hpp"
#include "grayscale_gpu.hpp"
#include "helpers_images.hpp"
#include "opening_gpu.hpp"
#include "utils_gpu.hpp"

void to_save(unsigned char *buffer_cuda, int rows, int cols, std::string file,
             size_t pitch)
{
    size_t size = rows * cols * sizeof(unsigned char);
    unsigned char *host_buffer =
        (unsigned char *)std::malloc(size * sizeof(unsigned char));

    cudaMemcpy2D(host_buffer, cols * sizeof(unsigned char), buffer_cuda, pitch,
                 cols * sizeof(unsigned char), rows, cudaMemcpyDeviceToHost);

    save(host_buffer, cols, rows, file);
    free(host_buffer);
}

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels)
{
    std::string file_save_gray_ref = "../images/gray_scale_ref_cuda.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj_cuda.jpg";

    std::string file_save_blur_ref = "../images/blurred_ref_cuda.jpg";
    std::string file_save_blur_obj = "../images/blurred_obj_cuda.jpg";

    std::string file_save_diff = "../images/diff_cuda.jpg";

    std::string file_save_closing_obj = "../images/closing_cuda.jpg";
    std::string file_save_opening_obj = "../images/opening_cuda.jpg";

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
    unsigned char *buffer_ref_cuda =
        cpy_host_to_device<unsigned char>(buffer_ref, size_color);
    unsigned char *buffer_obj_cuda =
        cpy_host_to_device<unsigned char>(buffer_obj, size_color);

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
    to_save(gray_obj_cuda, height, width, file_save_gray_obj, pitch);

    // Applying bluring
    unsigned int kernel_size = 5;
    double *kernel_gpu = create_gaussian_kernel_gpu(kernel_size);

    gaussian_blur_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        gray_ref_cuda, rows, cols, kernel_size, kernel_gpu, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();

    gaussian_blur_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        gray_obj_cuda, rows, cols, kernel_size, kernel_gpu, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();

    cudaFree(kernel_gpu);

    to_save(gray_ref_cuda, rows, cols, file_save_blur_ref, pitch);
    to_save(gray_obj_cuda, height, width, file_save_blur_obj, pitch);

    // Calculating diff
    difference<<<blocksPerGrid, threadsPerBlock>>>(gray_ref_cuda, gray_obj_cuda,
                                                   rows, cols, pitch);

    cudaCheckError();
    cudaDeviceSynchronize();

    unsigned char *current_obj = gray_obj_cuda;
    to_save(current_obj, rows, cols, file_save_diff, pitch);

    // Applying morphological opening/closing
    size_t k1_size = 5;
    size_t k2_size = 11;
    unsigned char *morpho_k1 = circular_kernel_gpu(k1_size);
    unsigned char *morpho_k2 = circular_kernel_gpu(k2_size);

    // Perform closing
    perform_erosion_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        current_obj, rows, cols, k1_size, morpho_k1, pitch);
    perform_dilation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        current_obj, rows, cols, k1_size, morpho_k1, pitch);

    cudaCheckError();
    cudaDeviceSynchronize();
    to_save(current_obj, rows, cols, file_save_closing_obj, pitch);

    // Perform opening
    perform_dilation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        current_obj, rows, cols, k2_size, morpho_k2, pitch);
    perform_erosion_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        current_obj, rows, cols, k2_size, morpho_k2, pitch);

    cudaCheckError();
    cudaDeviceSynchronize();

    to_save(current_obj, rows, cols, file_save_opening_obj, pitch);

    cudaFree(morpho_k1);
    cudaFree(morpho_k2);

    cudaFree(buffer_ref_cuda);
    cudaFree(buffer_obj_cuda);

    // Uncomment to see the results
    // to_save(gray_obj_cuda, height, width, file_save_diff, pitch);
}
