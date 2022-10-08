#include <iostream>
#include <vector>

void print(std::vector<std::vector<unsigned char>> input)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        auto line = input[i];
        for (size_t j = 0; j < line.size(); j++)
        {
            std::cout << (int) line[j];
        }
        std::cout << '\n';
    }
}

/*
Verify if the kernel is included in the image at the position (i, j)
*/
int intersection(std::vector<std::vector<unsigned char>> input, std::vector<std::vector<unsigned char>> kernel, int i, int j)
{
    size_t height_kernel = kernel.size();
    size_t width_kernel = kernel[0].size();
    size_t height = input.size();
    size_t width = input[0].size();
    int res = 0;

    if ((i + height_kernel) > height || (j + width_kernel) > width)
        return 0;

    for (size_t k = 0; k < height_kernel; k ++)
        for (size_t l = 0; l < width_kernel; l ++)
            if (input[k + i][l + j] != 0 && kernel[k][l] != 0)
                res += kernel[k][l];

    return res;
}

/*
Fill the output image with the values of the input image at the positions where
the kernel is equal to one
*/
std::vector<std::vector<unsigned char>> fill_opening(std::vector<std::vector<unsigned char>> input, std::vector<std::vector<unsigned char>> output, std::vector<std::vector<unsigned char>> kernel, int i, int j)
{
    size_t height_kernel = kernel.size();
    size_t width_kernel = kernel[0].size();
    size_t height = input.size();
    size_t width = input[0].size();

    if ((i + height_kernel) > height || (j + width_kernel) > width)
        return output;

    for (size_t k = 0; k < height_kernel; k ++)
        for (size_t l = 0; l < width_kernel; l ++)
            if (kernel[k][l] != 0)
                output[k + i][l + j] = input[k + i][l + j] * kernel[k][l];

    return output;
}


std::vector<std::vector<unsigned char>> fill_closing(std::vector<std::vector<unsigned char>> input, std::vector<std::vector<unsigned char>> output, std::vector<std::vector<unsigned char>> kernel, int i, int j)
{
    size_t height_kernel = kernel.size();
    size_t width_kernel = kernel[0].size();
    size_t height = input.size();
    size_t width = input[0].size();

    if ((i + height_kernel) > height || (j + width_kernel) > width)
        return output;

    for (size_t k = 0; k < height_kernel; k ++)
        for (size_t l = 0; l < width_kernel; l ++)
            if (kernel[k][l] != 0)
                output[k + i][l + j] = 0;

    return output;
}

/*
Creates a 2D vector filled with the value
*/
template  < typename  T >
std::vector<std::vector<T>> create_vector2D(size_t height, size_t width, T value)
{
    std::vector<std::vector<T>> res;
    for (size_t i = 0; i < height; i++)
    {
        std::vector<unsigned char> line(width, value);
        res.emplace_back(line);
    }
    return res;
}

/*
Sum the elements of a create_vector2D
*/
template  < typename  T >
int sum_vector(std::vector<std::vector<T>> vect)
{
    int res = 0;
    size_t height = vect.size();
    if (height == 0)
        return 0;
    size_t width = vect[0].size();
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            res += vect[i][j];
    return res;
}

/*
DeepCopy of a vector2D
*/
template  < typename  T >
std::vector<std::vector<T>> deep_copy(std::vector<std::vector<T>> vect)
{
    size_t height = vect.size();
    size_t width = vect[0].size();
    std::vector<std::vector<T>> res;
    for (size_t i = 0; i < height; i++)
    {
        std::vector<T> line;
        for (size_t j = 0; j < width; j++)
        {
            line.push_back(vect[i][j]);
        }
        res.emplace_back(line);
    }
    return res;
}

/*
DeepCopy of a vector2D
*/
template  < typename  T >
std::vector<std::vector<T>> add_padding(std::vector<std::vector<T>> vect, int pad_h, int pad_w)
{
    size_t height = vect.size();
    size_t width = vect[0].size();
    std::vector<std::vector<T>> res = create_vector2D<T>(height + 2 * pad_h, width + 2 * pad_w, 0);
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            res[i + pad_h][j + pad_w] = vect[i][j];

    return res;
}

/*
Compute the morphological opening of the image
*/
std::vector<std::vector<unsigned char>> perform_opening(std::vector<std::vector<unsigned char>> input, std::vector<std::vector<unsigned char>> kernel)
{
    size_t height = input.size();
    size_t width = input[0].size();
    int kernel_sum = sum_vector(kernel);
    auto output = create_vector2D<unsigned char>(height, width, 0);
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            if (intersection(input, kernel, i, j) == kernel_sum)
            {
                output = fill_opening(input, output, kernel, i, j);
            }
        }
    }
    return output;
}

/*
Compute the average
*/
int calculate_average(std::vector<std::vector<unsigned char>> mat, std::vector<std::vector<unsigned char>> kernel, int i, int j)
{
    size_t height_kernel = kernel.size();
    size_t width_kernel = kernel[0].size();
    int sum = 0;
    int quantity = 0;
    int left = - (width_kernel / 2);
    int top = - (height_kernel / 2);
    for (size_t k = 0; k < height_kernel; k++) {
        for (size_t l = 0; l < width_kernel; l++) {
            if (kernel[k][l] != 0 && mat[i + k + left][j + l + top] != 0)
            {
                quantity += 1;
                sum += mat[i + k + left][j + l + top];
            }

        }
    }
    return sum / quantity;
}

std::vector<std::vector<unsigned char>> mask_closing(std::vector<std::vector<unsigned char>> output, std::vector<std::vector<unsigned char>> mask, std::vector<std::vector<unsigned char>> kernel)
{
    size_t height = output.size();
    size_t width = output[0].size();
    size_t height_pad = kernel.size();
    size_t width_pad = kernel[0].size();
    for (size_t i = 0; i < (height); i++)
    {
        for (size_t j = 0; j < (width); j++)
        {
            if (mask[i + height_pad][j + width_pad] != 0 && output[i][j] == 0)
            {
                output[i][j] = calculate_average(output, kernel, i, j);
            }
        }
    }
    return output;
}

/*
Compute the morphological closing of the image
*/
std::vector<std::vector<unsigned char>> perform_closing(std::vector<std::vector<unsigned char>> input, std::vector<std::vector<unsigned char>> kernel)
{
    size_t height = input.size();
    size_t width = input[0].size();
    size_t height_kernel = kernel.size();
    size_t width_kernel = kernel[0].size();
    auto input_pad = add_padding<unsigned char>(input, height_kernel, width_kernel);
    auto mask = create_vector2D<unsigned char>(height + 2 * height_kernel, width + 2 * width_kernel, 1);
    auto output = deep_copy<unsigned char>(input);

    for (size_t i = 0; i < (height + 2 * height_kernel); i++)
    {
        for (size_t j = 0; j < (width + 2 * width_kernel); j++)
        {
            if (intersection(input_pad, kernel, i, j) == 0)
            {
                mask = fill_closing(input_pad, mask, kernel, i, j);
            }
        }
    }
    output = mask_closing(output, mask, kernel);
    return output;
}

int main()
{
    std::vector<std::vector<unsigned char>> input = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 5, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 5},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };
    std::vector<std::vector<unsigned char>> k1 = {{1,1,1}, {1,1,1}, {1,1,1}};
    std::vector<std::vector<unsigned char>> k2 = {{0,1,0}, {1,1,1}, {0,1,0}};
    auto output = perform_closing(input, k1);
    print(input);
    std::cout << '\n';
    print(output);
    return 0;
}
