#include <cassert>
#include <cmath>
#include <iostream>

#include "detect_obj_gpu.hpp"
#include "helpers_images.hpp"
#include "utils_gpu.hpp"

unsigned char *cpyToCuda(unsigned char *buffer_ref, size_t size) {
    return cpy_host_to_device<unsigned char>(buffer_ref, size); 
}

unsigned char *initCuda(size_t rows, size_t cols, size_t *pitch) {
    return malloc_cuda2D(cols, rows, pitch);
}

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

void object_detection(unsigned char *blur_ref_cuda,
                      unsigned char *img_obj,
                      unsigned char *buffer_obj_cuda,
                      size_t rows,
                      size_t cols,
                      size_t pitch,
                      int channels,
                      double *gaussian_k,
                      size_t gaussian_k_size,
                      unsigned char *morpho_k1,
                      size_t morpho_k1_size,
                      unsigned char *morpho_k2,
                      size_t morpho_k2_size,
                      int thx,
                      int thy) {

    const size_t size_color = cols * rows * channels * sizeof(unsigned char);
    unsigned char *img_obj_cuda = cpy_host_to_device<unsigned char>(img_obj, size_color);
    
    gray_scale_gpu(img_obj_cuda, buffer_obj_cuda, rows, cols, pitch, channels, thx, thy);
    apply_blurr_gpu(buffer_obj_cuda, rows, cols, gaussian_k_size, gaussian_k, pitch, thx, thy);
    difference_gpu(blur_ref_cuda, buffer_obj_cuda, rows, cols, pitch, thx, thy);
    
    unsigned char *current_obj = buffer_obj_cuda;
    
    erosion_gpu(current_obj, rows, cols, morpho_k1_size, morpho_k1, pitch, thx, thy);
    dilation_gpu(current_obj, rows, cols, morpho_k1_size, morpho_k1, pitch, thx, thy);

    dilation_gpu(current_obj, rows, cols, morpho_k2_size, morpho_k2, pitch, thx, thy);
    erosion_gpu(current_obj, rows, cols, morpho_k2_size, morpho_k2, pitch, thx, thy);
    
    unsigned char otsu_th = threshold(current_obj, rows, cols, pitch, thx, thy);
    int nb_components = connexe_components(current_obj, rows, cols, pitch, otsu_th, thx, thy);

    get_bbox(current_obj, rows, cols, pitch, nb_components);

    cudaFree(img_obj_cuda);
}

void main_detection_gpu(unsigned char** images,
                        int length,
                        int width,
                        int height,
                        int channels) {
    
    const int rows = height;
    const int cols = width;

    cudaDeviceProp deviceProp;
    int gpu_error = get_properties(0, &deviceProp);

    const dim3 threads(std::sqrt(deviceProp.maxThreadsDim[0]),
                       std::sqrt(deviceProp.maxThreadsDim[1]));
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                      std::ceil(float(rows) / float(threads.y)));

    const size_t size_color = cols * rows * channels * sizeof(unsigned char);

    // Copy content of host buffer to a cuda buffer
    unsigned char *buffer_ref_cuda = cpy_host_to_device<unsigned char>(images[0], size_color);

    size_t pitch; // we will store the pitch value in this variable
    unsigned char *gray_ref_cuda = malloc_cuda2D(cols, rows, &pitch);
    unsigned char *gray_obj_cuda = malloc_cuda2D(cols, rows, &pitch);

    gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, threads.x, threads.y);

    // Applying bluring
    unsigned int kernel_size = 5;
    double *kernel_gpu = create_gaussian_kernel_gpu(kernel_size);

    apply_blurr_gpu(gray_ref_cuda, rows, cols, kernel_size, kernel_gpu, pitch, threads.x, threads.y);
    
    size_t k1_size = 5;
    size_t k2_size = 11;
    unsigned char *morpho_k1 = circular_kernel_gpu(k1_size);
    unsigned char *morpho_k2 = circular_kernel_gpu(k2_size);

    for (int i = 1; i < length; i++) {
        object_detection(gray_ref_cuda, images[i], gray_obj_cuda, rows, cols, pitch,
                         channels, kernel_gpu, kernel_size, morpho_k1, k1_size, morpho_k2,
                         k2_size, threads.x, threads.y);
    }

    cudaFree(buffer_ref_cuda);
    cudaFree(gray_ref_cuda);
    cudaFree(gray_obj_cuda);
    cudaFree(kernel_gpu);
    cudaFree(morpho_k1);
    cudaFree(morpho_k2);
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
    std::string file_save_threshold_obj = "../images/threshold_cuda.jpg";

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

    gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, threadsPerBlock.x, threadsPerBlock.y);
    gray_scale_gpu(buffer_obj_cuda, gray_obj_cuda, rows, cols, pitch, channels, threadsPerBlock.x, threadsPerBlock.y);

    // Uncomment to see the images
    to_save(gray_ref_cuda, rows, cols, file_save_gray_ref, pitch);
    to_save(gray_obj_cuda, height, width, file_save_gray_obj, pitch);

    // Applying bluring
    unsigned int kernel_size = 5;
    double *kernel_gpu = create_gaussian_kernel_gpu(kernel_size);

    apply_blurr_gpu(gray_ref_cuda, rows, cols, kernel_size, kernel_gpu, pitch, threadsPerBlock.x, threadsPerBlock.y);
    apply_blurr_gpu(gray_obj_cuda, rows, cols, kernel_size, kernel_gpu, pitch, threadsPerBlock.x, threadsPerBlock.y);

    cudaFree(kernel_gpu);

    to_save(gray_ref_cuda, rows, cols, file_save_blur_ref, pitch);
    to_save(gray_obj_cuda, height, width, file_save_blur_obj, pitch);

    // Calculating diff
    difference_gpu(gray_ref_cuda, gray_obj_cuda, rows, cols, pitch, threadsPerBlock.x, threadsPerBlock.y);

    unsigned char *current_obj = gray_obj_cuda;
    to_save(current_obj, rows, cols, file_save_diff, pitch);

    // Applying morphological opening/closing
    size_t k1_size = 5;
    size_t k2_size = 11;
    unsigned char *morpho_k1 = circular_kernel_gpu(k1_size);
    unsigned char *morpho_k2 = circular_kernel_gpu(k2_size);

    // Perform closing
    erosion_gpu(current_obj, rows, cols, k1_size, morpho_k1, pitch, threadsPerBlock.x, threadsPerBlock.y);
    dilation_gpu(current_obj, rows, cols, k1_size, morpho_k1, pitch, threadsPerBlock.x, threadsPerBlock.y);

    to_save(current_obj, rows, cols, file_save_closing_obj, pitch);

    // Perform opening
    dilation_gpu(current_obj, rows, cols, k2_size, morpho_k2, pitch, threadsPerBlock.x, threadsPerBlock.y);
    erosion_gpu(current_obj, rows, cols, k2_size, morpho_k2, pitch, threadsPerBlock.x, threadsPerBlock.y);

    to_save(current_obj, rows, cols, file_save_opening_obj, pitch);

    cudaFree(morpho_k1);
    cudaFree(morpho_k2);

    unsigned char otsu_th = threshold(current_obj, rows, cols, pitch, threadsPerBlock.x, threadsPerBlock.y);
    to_save(current_obj, rows, cols, file_save_threshold_obj, pitch);

    int nb_components = connexe_components(current_obj, rows, cols, pitch, otsu_th, threadsPerBlock.x, threadsPerBlock.y);

    get_bbox(current_obj, rows, cols, pitch, nb_components);

    cudaFree(buffer_ref_cuda);
    cudaFree(buffer_obj_cuda);

    // Uncomment to see the results
    // to_save(gray_obj_cuda, height, width, file_save_diff, pitch);
}
