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
            size_t height, size_t width, size_t height_kernel,
            size_t width_kernel, bool dilation)
{
    int res = 0;

    int cst_height = (height_kernel - 1) / 2;
    int cst_width = (width_kernel - 1) / 2;
    for (size_t k = 0; k < height_kernel; k++)
    {
        for (size_t l = 0; l < width_kernel; l++)
        {
            if (kernel[k][l] == 0)
                continue;
            auto mul =
                input[k + i - cst_height][l + j - cst_width] * kernel[k][l];
            if (mul > res && dilation)
                res = mul;
            if (!dilation && (res == 0 || mul < res))
            {
                res = mul;
            }
        }
    }
    return res;
}

/*
Sum the elements of a create_array2D
*/
template <typename T>
int sum_vector(T **vect, size_t height, size_t width)
{
    int res = 0;
    if (height == 0)
        return 0;
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            res += vect[i][j];
    return res;
}

/*
DeepCopy of a vector2D
*/
template <typename T>
T **deep_copy(T **vect, size_t height, size_t width)
{
    T **res = (T **)malloc(height * sizeof(T *));
    for (size_t i = 0; i < height; i++)
    {
        T *line = (T *)malloc(width * sizeof(T));
        for (size_t j = 0; j < width; j++)
        {
            line[j] = vect[i][j];
        }
        res[i] = line;
    }
    return res;
}

/*
DeepCopy of a vector2D
*/
template <typename T>
T **add_padding(T **vect, size_t height, size_t width, int pad_h, int pad_w)
{
    T **res = create_array2D<T>(height + 2 * pad_h, width + 2 * pad_w, 0);
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            res[i + pad_h][j + pad_w] = vect[i][j];

    return res;
}

/*
Compute the morphological opening of the image
*/
unsigned char **perform_dilation(unsigned char **input, unsigned char **kernel,
                                 size_t height, size_t width,
                                 size_t height_kernel, size_t width_kernel)
{
    int kernel_sum = sum_vector(kernel, height_kernel, width_kernel);
    auto output = create_array2D<unsigned char>(height, width, 0);
    int cst_height = (height_kernel - 1) / 2;
    int cst_width = (width_kernel - 1) / 2;
    for (size_t i = cst_height; i < height - cst_height; i++)
    {
        for (size_t j = cst_width; j < width - cst_width; j++)
        {
            output[i][j] = product(input, kernel, i, j, height, width,
                                   height_kernel, width_kernel, true);
        }
    }
    return output;
}

unsigned char **perform_erosion(unsigned char **input, unsigned char **kernel,
                                size_t height, size_t width,
                                size_t height_kernel, size_t width_kernel)
{
    auto output = create_array2D<unsigned char>(height, width, 0);
    int cst_height = (height_kernel - 1) / 2;
    int cst_width = (width_kernel - 1) / 2;
    for (size_t i = cst_height; i < height - cst_height; i++)
    {
        for (size_t j = cst_width; j < width - cst_width; j++)
        {
            output[i][j] = product(input, kernel, i, j, height, width,
                                   height_kernel, width_kernel, false);
        }
    }
    return output;
}

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
