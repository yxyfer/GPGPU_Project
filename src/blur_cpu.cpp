#include "detect_obj.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

// (1 / 2*pi*sigma^2) / e(-(x^2 + y^2)/2 * sigma^2)
struct GaussianKernel* create_gaussian_kernel(unsigned char size) {
    double **kernel = create2Dmatrix<double>(size, size);

    int margin = (int) size / 2;
    double sigma = 1.0;
    double s = 2.0 * sigma * sigma;
    
    // sum is for normalization
    double sum = 0.0;

    for (int row = -margin; row <= margin; row++) {
        for (int col = -margin; col <= margin; col++) {
            const double radius = col * col + row * row;
            kernel[row + margin][col + margin] = (exp(-radius / s)) / (M_PI * s);
            sum += kernel[row + margin][col + margin];
        }
    }

    // normalising the Kernel
    for (unsigned char i = 0; i < size; ++i)
        for (unsigned char j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    struct GaussianKernel *g_kernel = (struct GaussianKernel *)std::malloc(sizeof(struct GaussianKernel));
    g_kernel->kernel = kernel;
    g_kernel->size = size;

    return g_kernel;
}

unsigned char convolution(int x, int y, struct ImageMat *image, struct GaussianKernel* kernel)
{
    double conv = 0;
    // get kernel val
    int margin = (int) kernel->size / 2;
    for(int i = -margin ; i <= margin; i++)
        for(int j = -margin; j <= margin; j++) {
            if (x + i < 0 || y + j < 0 || x + i >= image->height || y + j >= image->width)
                continue;
            conv += image->pixel[x + i][y + j] * kernel->kernel[i + margin][j + margin];
        }

    return conv;
}

void apply_blurring(struct ImageMat *image, struct ImageMat *temp, struct GaussianKernel *kernel) {
    for (int x = 0; x < image->height; ++x)
        for (int y = 0; y < image->width; ++y) {
            temp->pixel[x][y] = convolution(x, y, image, kernel);
        }

    swap_matrix(image, temp);
}

