#include <cmath>
#include <iostream>
#include <vector>

#include "detect_obj.hpp"
#include "utils.hpp"

void print_mat(unsigned char **input, size_t height, size_t width)
{
    for (size_t i = 0; i < height; i++)
    {
        auto line = input[i];
        for (size_t j = 0; j < width; j++)
        {
            std::cout << (int)line[j];
        }
        std::cout << '\n';
    }
}

/*
Verify if the kernel is included in the image at the position (i, j)
*/
int product(unsigned char **input, unsigned char **kernel, int i, int j,
            size_t kernel_size, bool dilation)
{
    int res = 0;

    int cst_size = (kernel_size - 1) / 2;
    for (size_t k = 0; k < kernel_size; k++)
    {
        for (size_t l = 0; l < kernel_size; l++)
        {
            if (kernel[k][l] == 0)
                continue;

            auto mul = input[k + i - cst_size][l + j - cst_size] * kernel[k][l];

            if (mul > res && dilation)
                res = mul;
            
            if (!dilation && (res == 0 || mul < res))
                res = mul;
        }
    }
    return res;
}

/*
Compute the morphological opening of the image
*/
void perform_dilation(unsigned char **input, unsigned char **temp, unsigned char **kernel,
                                 size_t height, size_t width, size_t kernel_size)
{
    int cst_size = (kernel_size - 1) / 2;

    for (size_t i = cst_size; i < height - cst_size; i++)
    {
        for (size_t j = cst_size; j < width - cst_size; j++)
        {
            temp[i][j] = product(input, kernel, i, j, kernel_size, true);
        }
    }
}

void perform_erosion(unsigned char **input, unsigned char **temp, unsigned char **kernel,
                                size_t height, size_t width, size_t kernel_size)
{
    int cst_size = (kernel_size - 1) / 2;

    for (int i = 0; i < cst_size; i++) {
        for (int j = 0; j < cst_size; j++) {
            temp[i][j] = 0;
            temp[height - i - 1][width - j - 1] = 0;
        }
    }

    for (size_t i = cst_size; i < height - cst_size; i++)
    {
        for (size_t j = cst_size; j < width - cst_size; j++)
        {
            temp[i][j] = product(input, kernel, i, j, kernel_size, false); 
        }
    }
}

/* unsigned char **perform_opening(unsigned char **input, unsigned char **kernel, */
/*                                 size_t height, size_t width, size_t kernel_size) { */
/*     auto opening_dil = perform_dilation(input, kernel, height, width, kernel_size); */
/*     auto opening_ero = perform_erosion(opening_dil, kernel, height, width, kernel_size); */

/*     free2Dmatrix(height, opening_dil); */
/*     return opening_ero; */
/* } */

/* unsigned char **perform_closing(unsigned char **input, unsigned char **kernel, */
/*                                 size_t height, size_t width, size_t kernel_size) { */
/*     auto closing_ero = perform_erosion(input, kernel, height, width, kernel_size); */
/*     auto closing_dil = perform_dilation(closing_ero, kernel, height, width, kernel_size); */

/*     free2Dmatrix(height, closing_ero); */
/*     return closing_dil; */
/* } */

// kernel_size must be odd
unsigned char **circular_kernel(int kernel_size)
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
    return kernel;
}
