#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

struct Bbox
{
    int x;
    int y;
    int height;
    int width;
};

unsigned char *cpyToCuda(unsigned char *buffer_ref, size_t size);
unsigned char *initCuda(size_t rows, size_t cols, size_t *pitch);

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels);

struct Bbox ***main_detection_gpu(unsigned char **images, int length, int width, int height,
                                  int channels, int *nb_objs);

void gray_scale_gpu(unsigned char *buffer_cuda, unsigned char *gray_cuda,
                    int rows, int cols, int pitch, int channels, int thx, int thy);

void gray_scale_test(unsigned char *buffer, int width, int height, int channels);

double *create_gaussian_kernel_gpu(unsigned char size);

void apply_blurr_gpu(unsigned char *buffer, size_t rows, size_t cols, unsigned int kernel_size,
                     double *kernel_gpu, size_t pitch, int thx, int thy);

void difference_gpu(unsigned char *ref, unsigned char *obj, size_t rows,
                    size_t cols, size_t pitch, int thx, int thy);

unsigned char *circular_kernel_gpu(int kernel_size);

void erosion_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                 unsigned char *kernel, size_t pitch, int thx, int thy);

void dilation_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                 unsigned char *kernel, size_t pitch, int thx, int thy);

unsigned char otsu_threshold(unsigned char* buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

unsigned char threshold(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int thx, int thy);

int connexe_components(unsigned char *buffer_base, size_t rows, size_t cols, size_t pitch, unsigned char threshold, int thx, int thy);

struct Bbox **get_bbox(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int nb_components);

