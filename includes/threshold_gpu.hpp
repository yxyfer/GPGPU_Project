#pragma once
#include <cstddef>
#include <memory>

int test_function(unsigned char* buffer, size_t rows, size_t cols);

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch);

void threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch);

