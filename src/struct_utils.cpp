#include "utils.hpp"
#include "detect_obj.hpp"

#include <cmath>

void swap_matrix(struct ImageMat *a, struct ImageMat *b) {
    unsigned char **temp = a->pixel;
    a->pixel = b->pixel;
    b->pixel = temp;
}

struct ImageMat *new_matrix(int height, int width) {
    struct ImageMat *matrix = (struct ImageMat *)std::malloc(sizeof(struct ImageMat));
    matrix->pixel = create2Dmatrix<unsigned char>(height, width);
    matrix->height = height;
    matrix->width = width;

    return matrix;
}

// kernel_size must be odd
struct MorphologicalKernel* circular_kernel(int kernel_size)
{
    auto kernel = create_array2D<unsigned char>(kernel_size, kernel_size, 0);
    int radius = kernel_size / 2;
    for (int x = -radius; x < radius + 1; x++)
    {
        int y = (std::sqrt(radius * radius - (x * x)));
        for (int j = -y; j < y + 1; j++)
        {
            kernel[j + radius][x + radius] = 1;
            kernel[-j + radius][x + radius] = 1;
        }
    }

    auto c_kernel = (struct MorphologicalKernel *)std::malloc(sizeof(struct MorphologicalKernel *));
    c_kernel->kernel = kernel;
    c_kernel->size = kernel_size;

    return c_kernel;
}

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
