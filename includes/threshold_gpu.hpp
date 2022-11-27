#pragma once
#include <cstddef>
#include <memory>

int test_function(unsigned char* buffer, size_t rows, size_t cols);

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

void threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

void connexe_components(unsigned char *buffer_base, unsigned int *buffer_bin,
                        size_t rows, size_t cols, size_t pitch, size_t pitch_bin, int thx, int thy);

