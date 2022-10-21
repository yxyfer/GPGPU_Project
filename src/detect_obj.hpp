#pragma once
#include <cstddef>
#include <memory>

struct ImageMat {
    unsigned char **pixel;
    int height;
    int width;
};

struct GaussianKernel {
    double **kernel;
    unsigned char size;
};

void swap_matrix(struct ImageMat *a, struct ImageMat *b);

// FILE: detect_obj_cpu.cpp
/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
unsigned char** detect_cpu(unsigned char* buffer_ref,
                           unsigned char* buffer_obj,
                           int width,
                           int height,
                           int channels);

// FILE: detect_obj_cpu.cpp
/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void to_gray_scale(unsigned char* src,
                   struct ImageMat* dst,
                   int width,
                   int height,
                   int channels);

struct GaussianKernel* create_gaussian_kernel(unsigned char size);

// FILE: detect_obj_cpu.cpp
/// \param gray_ref The gray scale image buffer
/// \param gray_obj The gray scale image buffer
/// \param width Image width
/// \param height Image height
void difference(struct ImageMat *ref, struct ImageMat *obj);


// FILE: blur_cpu.cpp
// Apply gaussian blur to the image
/// \param image The gray scale image
/// \param width Image width
/// \param height Image height
/// \param kernel_size Kernel size
void apply_blurring(struct ImageMat* image,
                    struct ImageMat* temp,
                    struct GaussianKernel* kernel);

void perform_dilation(unsigned char** input,
                      unsigned char** temp,
                      unsigned char** kernel,
                      size_t height,
                      size_t width,
                      size_t kernel_size);

void perform_erosion(unsigned char** input,
                     unsigned char** temp,
                     unsigned char** kernel,
                     size_t height,
                     size_t width,
                     size_t kernel_size);

unsigned char** perform_opening(unsigned char** input,
                                 unsigned char** kernel,
                                 size_t height,
                                 size_t width,
                                 size_t kernel_size);

unsigned char** perform_closing(unsigned char** input,
                                unsigned char** kernel,
                                size_t height,
                                size_t width,
                                size_t kernel_size);

unsigned char** compute_threshold(unsigned char** image, int width, int height);

/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_gpu(unsigned char* buffer_ref,
                unsigned char* buffer_obj,
                int width,
                int height,
                int channels);

unsigned char** circular_kernel(int kernel_size);

void print_mat(unsigned char** input, size_t height, size_t width);
