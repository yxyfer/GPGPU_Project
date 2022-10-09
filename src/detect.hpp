#pragma once
#include <cstddef>
#include <memory>




/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels);

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
unsigned char **to_gray_scale(unsigned char *buffer, int width, int height, int channels);

/// \param gray_ref The gray scale image buffer
/// \param gray_obj The gray scale image buffer
/// \param width Image width
/// \param height Image height
unsigned char **difference(unsigned char **gray_ref, unsigned char **gray_obj, int width, int height);


/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels);
