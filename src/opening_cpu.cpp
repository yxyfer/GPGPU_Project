#include <cmath>
#include <iostream>

#include "detect_obj.hpp"

/*
Verify if the kernel is included in the image at the position (i, j)
*/

int product(struct ImageMat *input, struct MorphologicalKernel *kernel, int i,
            int j, bool dilation)
{
    int res = 0;

    int cst_size = (kernel->size - 1) / 2;
    for (int k = 0; k < kernel->size; k++)
    {
        if ((k + i - cst_size < 0) || (k + i - cst_size >= input->height))
            continue;

        for (int l = 0; l < kernel->size; l++)
        {
            if ((kernel->kernel[k][l] == 0) || (l + j - cst_size < 0)
                || (l + j - cst_size >= input->width))
                continue;

            const auto mul = input->pixel[k + i - cst_size][l + j - cst_size]
                * kernel->kernel[k][l];

            if (dilation && mul > res)
                res = mul;

            if (!dilation && (res == 0 || mul < res))
                res = mul;
        }
    }
    return res;
}

void perform_dilation(struct ImageMat *input, struct ImageMat *temp,
                      struct MorphologicalKernel *kernel)
{
    for (int i = 0; i < input->height; i++)
        for (int j = 0; j < input->width; j++)
            temp->pixel[i][j] = product(input, kernel, i, j, true);

    swap_matrix(input, temp);
}

void perform_erosion(struct ImageMat *input, struct ImageMat *temp,
                     struct MorphologicalKernel *kernel)
{
    for (int i = 0; i < input->height; i++)
        for (int j = 0; j < input->width; j++)
            temp->pixel[i][j] = product(input, kernel, i, j, false);

    swap_matrix(input, temp);
}
