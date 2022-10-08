#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <png.h>

unsigned char convolution(int x, int y, char **mat, char** kernel, int kernel_size)
{
    unsigned int conv = 0;
    for(int i = 0; x + i < x + kernel_size; i++)
        for(int j = 0; x + j < y + kernel_size; j++)
            conv += mat[x + i][x + j] * kernel[i][j];
    return conv;
}

unsigned char gb_convolution(int x, int y, char **mat, char **kernel, int kernel_size)
{
    unsigned int sum = 0;
    for(int i = 0; i < kernel_size; i++)
        for(int j = 0; j < kernel_size; j++)
            sum += kernel[i][j];
    return convolution(x, y, mat, kernel, kernel_size) / sum; 
}

int main()
{
    int kernel_size = 3;
    int matrix_size = 3;
    char **mat = (char **) malloc(matrix_size * sizeof(char *));
    for(int i = 0; i < matrix_size; i++)
        mat[i] = (char *) malloc(matrix_size * sizeof(char));

    for(int i = 0; i < matrix_size; i++)
    {
        for(int j = 0; j < matrix_size; j++)
        {
            if (i == 0)
                mat[i][j] = 1 + 2 * j;
            else
                mat[i][j] = 1 + j;
        }
    }

    char **gaussian_blur = (char **) malloc(kernel_size * sizeof(char *));
    for(int i = 0; i < matrix_size; i++)
        gaussian_blur[i] = (char *) malloc(kernel_size * sizeof(char));

    for(int i = 0; i < kernel_size; i++)
        for(int j = 0; j < kernel_size; j++)
            gaussian_blur[i][j] = 1 * (i == 1 ? 2 : 1) * (j == 1 ? 2 : 1);

    unsigned char conv = gb_convolution(0, 0, mat, gaussian_blur, kernel_size);
    std::cout << (int) conv << std::endl;
    return 0;
}
