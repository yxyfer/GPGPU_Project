#pragma once
#include <cstddef>
#include <memory>

int test_function(unsigned char* buffer, size_t rows, size_t cols);

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

unsigned char threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

int connexe_components(unsigned char *buffer_base, size_t rows, size_t cols, size_t pitch, unsigned char threshold, int thx, int thy);

