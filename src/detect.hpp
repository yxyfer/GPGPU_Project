#pragma once
#include <cstddef>
#include <memory>




/// \param buffer_start The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
void detect_cpu(char* buffer_start, char *buffer_obj, int width, int height, std::ptrdiff_t stride);



/// \param buffer_start The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
void detect_gpu(char* buffer_start, char *buffer_obj, int width, int height, std::ptrdiff_t stride);
