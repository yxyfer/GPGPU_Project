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
