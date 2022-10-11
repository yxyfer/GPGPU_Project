#pragma once
#include <cstddef>
#include <memory>

// FILE: detect_obj_cpu.cpp
/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels);

// FILE: detect_obj_cpu.cpp
/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
unsigned char **to_gray_scale(unsigned char *buffer, int width, int height, int channels);

// FILE: detect_obj_cpu.cpp
/// \param gray_ref The gray scale image buffer
/// \param gray_obj The gray scale image buffer
/// \param width Image width
/// \param height Image height
unsigned char **difference(unsigned char **gray_ref, unsigned char **gray_obj, int width, int height);

// FILE: blur_cpu.cpp
// Apply gaussian blur to the image
/// \param image The gray scale image
/// \param width Image width
/// \param height Image height
/// \param kernel_size Kernel size
unsigned char **apply_blurring(unsigned char** image, int width, int height, unsigned char kernel_size);


unsigned char **perform_dilation(unsigned char **input, unsigned char **kernel,
                                 size_t height, size_t width,
                                 size_t height_kernel, size_t width_kernel);

unsigned char **perform_erosion(unsigned char **input, unsigned char **kernel,
                                size_t height, size_t width,
                                size_t height_kernel, size_t width_kernel);

unsigned char **perform_opening(unsigned char **input, unsigned char **kernel,
                                size_t height, size_t width,
                                size_t height_kernel, size_t width_kernel);


/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels);
