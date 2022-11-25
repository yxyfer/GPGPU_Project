#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

void detect_gpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width,
                int height, int channels);

/*void apply_gaussian_blur_gpu(unsigned char *gray_ref_cuda,
                             unsigned char *gray_obj_cuda, int rows, int cols,
                             int kernel_size, kernelCudaSize cuda_kernel_size,
                             int pitch);*/
