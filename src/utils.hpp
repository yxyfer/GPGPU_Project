#pragma once
#include <iostream>


template <typename T>
T **create2Dmatrix(size_t rows, size_t cols) {
    T** matrix = (T**) malloc(rows * sizeof(T*));
    for(size_t i = 0; i < rows; i++)
        matrix[i] = (T*) malloc(cols * sizeof(T));

    return matrix;
}


template <typename T>
T **create_array2D(size_t height, size_t width, T value)
{
    T **res = (T **)malloc(height * sizeof(T *));
    for (size_t i = 0; i < height; i++)
    {
        T *line = (T *)malloc(width * sizeof(T));
        for (size_t j = 0; j < width; j++)
        {
            line[j] = value;
        }
        res[i] = line;
    }
    return res;
}


template <typename T>
void free2Dmatrix(size_t rows, T matrix) {
    for (size_t i = 0; i < rows; i++)
        free(matrix[i]);

    free(matrix);
}


/*
Sum the elements of a create_array2D
*/
/* template <typename T> */
/* int sum_vector(T **vect, size_t height, size_t width) */
/* { */
/*     int res = 0; */
/*     if (height == 0) */
/*         return 0; */
/*     for (size_t i = 0; i < height; i++) */
/*         for (size_t j = 0; j < width; j++) */
/*             res += vect[i][j]; */
/*     return res; */
/* } */
