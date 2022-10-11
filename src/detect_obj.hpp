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
/// \param kernel The gaussian blur kernel
/// \param width Image width
/// \param height Image height
/// \param kernel_size Kernel size
unsigned char **blurring(unsigned char** image, unsigned char** kernel, int width, int height, unsigned char kernel_size); 



/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels);
