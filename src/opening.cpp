#include <iostream>
#include <vector>

void print(unsigned char **input, size_t height, size_t width)
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
            auto mul =
                input[k + i - cst_height][l + j - cst_width] * kernel[k][l];
            if (kernel[k][l] == 0)
                continue;
            if (mul > res && dilation)
                res = mul;
            if (!dilation && (res == 0 || mul < res))
            {
                res = mul;
            }
        }
    }
    /*
        if (input[k + i][l + j] != 0 && kernel[k][l] != 0)
            res += kernel[k][l];*/
    return res;
}

/*
Fill the output image with the values of the input image at the positions where
the kernel is equal to one
*/
unsigned char **fill_opening(unsigned char **input, unsigned char **output,
                             unsigned char **kernel, int i, int j,
                             size_t height, size_t width, size_t height_kernel,
                             size_t width_kernel)
{
    if ((i + height_kernel) > height || (j + width_kernel) > width)
        return output;

    for (size_t k = 0; k < height_kernel; k++)
        for (size_t l = 0; l < width_kernel; l++)
            if (kernel[k][l] != 0)
                output[k + i][l + j] = input[k + i][l + j] * kernel[k][l];

    return output;
}

unsigned char **fill_closing(unsigned char **input, unsigned char **output,
                             unsigned char **kernel, int i, int j,
                             size_t height, size_t width, size_t height_kernel,
                             size_t width_kernel)
{
    if ((i + height_kernel) > height || (j + width_kernel) > width)
        return output;

    for (size_t k = 0; k < height_kernel; k++)
        for (size_t l = 0; l < width_kernel; l++)
            if (kernel[k][l] != 0)
                output[k + i][l + j] = 0;

    return output;
}

/*
Creates a 2D vector filled with the value
*/
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
            /*
            if (intersection(input, kernel, i, j, height, width, height_kernel,
                             width_kernel)
                == kernel_sum)
            {
                output = fill_opening(input, output, kernel, i, j, height,
                                      width, height_kernel, width_kernel);
            }*/
        }
    }
    return output;
}

unsigned char **perform_erosion(unsigned char **input, unsigned char **kernel,
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
                                   height_kernel, width_kernel, false);
            /*
            if (intersection(input, kernel, i, j, height, width, height_kernel,
                             width_kernel)
                == kernel_sum)
            {
                output = fill_opening(input, output, kernel, i, j, height,
                                      width, height_kernel, width_kernel);
            }*/
        }
    }
    return output;
}

/*
Compute the average
*/
int calculate_average(unsigned char **mat, unsigned char **kernel, int i, int j,
                      size_t height_kernel, size_t width_kernel)
{
    int sum = 0;
    int quantity = 0;
    int left = -(width_kernel / 2);
    int top = -(height_kernel / 2);
    for (size_t k = 0; k < height_kernel; k++)
    {
        for (size_t l = 0; l < width_kernel; l++)
        {
            if (kernel[k][l] != 0 && mat[i + k + left][j + l + top] != 0)
            {
                quantity += 1;
                sum += mat[i + k + left][j + l + top];
            }
        }
    }
    return sum / quantity;
}

unsigned char **mask_closing(unsigned char **output, unsigned char **mask,
                             unsigned char **kernel, size_t height,
                             size_t width, size_t height_kernel,
                             size_t width_kernel)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            if (mask[i + height_kernel][j + width_kernel] != 0
                && output[i][j] == 0)
            {
                output[i][j] = calculate_average(output, kernel, i, j,
                                                 height_kernel, width_kernel);
            }
        }
    }
    return output;
}

/*
Compute the morphological closing of the image
*/
/*
unsigned char **perform_closing(unsigned char **input, unsigned char **kernel,
                                size_t height, size_t width,
                                size_t height_kernel, size_t width_kernel)
{
    auto input_pad = add_padding<unsigned char>(input, height, width,
                                                height_kernel, width_kernel);
    auto mask = create_array2D<unsigned char>(height + 2 * height_kernel,
                                              width + 2 * width_kernel, 1);
    auto output = deep_copy<unsigned char>(input, height, width);

    for (size_t i = 0; i < (height + 2 * height_kernel); i++)
    {
        for (size_t j = 0; j < (width + 2 * width_kernel); j++)
        {
            if (product(input_pad, kernel, i, j, height, width, height_kernel,
                        width_kernel)
                == 0)
            {
                mask = fill_closing(input_pad, mask, kernel, i, j, height,
                                    width, height_kernel, width_kernel);
            }
        }
    }
    output = mask_closing(output, mask, kernel, height, width, height_kernel,
                          width_kernel);
    return output;
}
*/
int main()
{
    unsigned char **input = create_array2D<unsigned char>(11, 29, 0);
    unsigned char input_t[11][29] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
          0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
          0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
          5, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 5 },
        { 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    for (size_t i = 0; i < 11; i++)
    {
        for (size_t j = 0; j < 29; j++)
        {
            input[i][j] = input_t[i][j];
        }
    }
    unsigned char **k1 = create_array2D<unsigned char>(3, 3, 1);
    unsigned char **k2 = create_array2D<unsigned char>(3, 3, 0);
    unsigned char k2_t[3][3] = { { 0, 1, 0 }, { 1, 1, 1 }, { 0, 1, 0 } };
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            k2[i][j] = k2_t[i][j];
        }
    }
    auto output = perform_dilation(input, k2, 11, 29, 3, 3);
    output = perform_erosion(output, k2, 11, 29, 3, 3);
    // output = perform_dilation(output, k1, 11, 29, 3, 3);
    // output = perform_erosion(output, k1, 11, 29, 3, 3);
    print(input, 11, 29);
    std::cout << '\n';
    print(output, 11, 29);

    return 0;
}
