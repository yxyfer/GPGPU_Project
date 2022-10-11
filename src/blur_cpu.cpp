#include "detect_obj.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

// (1 / 2*pi*sigma^2) / e(-(x^2 + y^2)/2 * sigma^2)
double **create_gaussian_kernel(unsigned char size) {
    double **kernel = create2Dmatrix<double>(size, size);
    int margin = (int) size / 2;
    double sigma = 3.0;

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

    return kernel;
}

unsigned char convolution(int x, int y, int height, int width, unsigned char **image, double** kernel, unsigned char kernel_size)
{
    double conv = 0;
    // get kernel val
    int margin = (int) kernel_size / 2;
    for(int i = -margin ; i <= margin; i++)
        for(int j = -margin; j <= margin; j++) {
            if (x + i < 0 || y + j < 0 || x + i >= height || y + j >= width)
                continue;
            conv += image[x + i][y + j] * kernel[i + margin][j + margin];
        }

    return conv;
}

unsigned char **apply_blurring(unsigned char** image, int width, int height, unsigned char kernel_size) {
    // Allocating for blurred image
    unsigned char** blurred_image = create2Dmatrix<unsigned char>(height, width);
    double **kernel = create_gaussian_kernel(kernel_size);
    
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            blurred_image[x][y] = convolution(x, y, height, width, image, kernel, kernel_size);
        }
    
   return blurred_image; 
}

