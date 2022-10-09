#pragma once
#include <cstddef>
#include <memory>




/// \param buffer_start The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_cpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels);

/// \param buffer_start The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
unsigned char **to_gray_scale(unsigned char* buffer, int width, int height, int channels);


/// \param buffer_start The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
void detect_gpu(unsigned char* buffer_start, unsigned char *buffer_obj, int width, int height, int channels);
