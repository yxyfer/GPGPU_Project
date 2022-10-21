#include "detect_obj.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>


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

