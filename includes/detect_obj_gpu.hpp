#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

unsigned char *cpyToCuda(unsigned char *buffer_ref, size_t size);
unsigned char *initCuda(size_t rows, size_t cols, size_t *pitch);

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels);

void gray_scale_gpu(unsigned char *buffer_cuda, unsigned char *gray_cuda,
                    int rows, int cols, int pitch, int channels, int thx, int thy);

void gray_scale_test(unsigned char *buffer, int width, int height, int channels);


void get_bbox(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int nb_components);

