#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <limits>

#include "utils.hpp"

float inf = std::numeric_limits<float>::infinity();

float var(unsigned char* array, int nb_pixels)
{
    float sum = 0.0, mean, variance = 0.0;

    for (int i = 0; i < nb_pixels; ++i)
        sum += array[i];

    mean = (float)sum / (float)nb_pixels;

    // TODO: NOT OPTIMISED WHAT SO EVER!
    for (int i = 0; i < nb_pixels; ++i)
        variance += pow((float)array[i] - mean, 2);

    variance /= (float)nb_pixels;

    return variance;
}

float otsu_criteria(unsigned char** image,
                    unsigned char threshold,
                    int width,
                    int height)
{
    unsigned char** thresholded_image =
        create2Dmatrix<unsigned char>(height, width);

    unsigned int nb_pixels = width * height;
    unsigned int nb_whitep = 0;

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (image[x][y] >= threshold) {
                thresholded_image[x][y] = 1;
                nb_whitep++;
            }
            else
                thresholded_image[x][y] = 0;
        }

    // Seperate white and black pixels into two arrays
    unsigned char* white_pixels =
        (unsigned char*)malloc(nb_whitep * sizeof(unsigned char));
    unsigned char* black_pixels =
        (unsigned char*)malloc((nb_pixels - nb_whitep) * sizeof(unsigned char));

    unsigned int wp_i = 0;
    unsigned int bp_i = 0;

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (thresholded_image[x][y] == 1) {
                white_pixels[wp_i] = image[x][y];
                wp_i++;
            }
            else {
                black_pixels[bp_i] = image[x][y];
                bp_i++;
            }
        }

    float var_white = var(white_pixels, nb_whitep);
    float var_black = var(black_pixels, nb_pixels - nb_whitep);

    // Finish calc

    float weight_whitep = (float)nb_whitep / (float)nb_pixels;
    float weight_blackp = 1 - weight_whitep;

    // printf("\n\ncurr threshold %d\n", threshold);
    // printf("\nnb white p %d", nb_whitep);
    // printf("\nnb pixels: %d", nb_pixels);
    // printf("\nvar white %d", var_white);
    // printf("\nwheight blackp %f", weight_blackp);
    // printf("\nweight white p %f", weight_whitep);

    if (weight_whitep == 0 || weight_blackp == 0)
        return inf;

    return (float)weight_whitep * var_white + (float)weight_blackp * var_black;
}

unsigned char get_max(unsigned char* array, int size)
{
    unsigned char max = 0;
    for (int i = 0; i < size; ++i) {
        max = max < array[i] ? array[i] : max;
    }

    return max;
}

int get_min_index(float* array, int size)
{
    int min = 0;
    float one = array[min];

    for (int i = 0; i < size; ++i) {
        float curr = array[i];
        if (curr < one && curr > 0) {
            // printf("%f < %f ", curr, one);
            min = i;
            one = array[min];
        }
    }
    return min;
}

int get_max_pixel_val(unsigned char** in_image, int array_size)
{
    // Getting max values in image
    unsigned char max_pixel_vals[array_size];
    for (int x = 0; x < array_size; ++x)
        max_pixel_vals[x] = get_max(in_image[x], array_size);
    int max_pixel_val = get_max(max_pixel_vals, array_size);

    return max_pixel_val;
}

unsigned char get_otsu_threshold(unsigned char** in_image,
                                 int width,
                                 int height)
{
    // getting max image pixel value
    int max_pixel_val = get_max_pixel_val(in_image, height);

    int range_size = max_pixel_val + 1;
    unsigned char threshold_range[range_size];
    for (int i = 1; i < range_size; ++i)
        threshold_range[i - 1] = i;

    float otsu_vals[range_size];
    for (int i = 0; i < range_size; ++i) {
        unsigned char curr_threshold = threshold_range[i];
        otsu_vals[i] = otsu_criteria(in_image, curr_threshold, width, height);
    }

    int threshold_index = get_min_index(otsu_vals, range_size);
    unsigned char otsu_threshold = threshold_range[threshold_index];

    printf("\nmin index %d ", threshold_index);
    printf("\nthreshold %d ", otsu_threshold);

    return otsu_threshold;
}

unsigned char** apply_base_threshold(unsigned char** in_image,
                                     unsigned char threshold,
                                     int width,
                                     int height)
{
    unsigned char** base_image = create2Dmatrix<unsigned char>(height, width);

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (in_image[x][y] >= threshold)
                base_image[x][y] = in_image[x][y];
            else
                base_image[x][y] = 0;
        }

    return base_image;
}
unsigned char** apply_bin_threshold(unsigned char** in_image,
                                    unsigned char threshold,
                                    int width,
                                    int height)
{
    unsigned char** bin_image = create2Dmatrix<unsigned char>(height, width);

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (in_image[x][y] >= threshold)
                bin_image[x][y] = 255;
            else
                bin_image[x][y] = 0;
        }

    return bin_image;
}

unsigned char** compute_otsu_threshold(unsigned char** in_image,
                                       int width,
                                       int height)
{
    unsigned char otsu_threshold = get_otsu_threshold(in_image, width, height);
    unsigned char** base_image =
        apply_base_threshold(in_image, otsu_threshold, width, height);

    unsigned char otsu_threshold2 =
        get_otsu_threshold(base_image, width, height);
    unsigned char** bin_image =
        apply_bin_threshold(base_image, otsu_threshold2, width, height);

    printf("otsu_threshold 1 %i", otsu_threshold);
    printf("otsu_threshold 2 %i", otsu_threshold2);

    return bin_image;
}

unsigned char** compute_threshold(unsigned char** image, int width, int height)
{
    unsigned char** thresholded_image =
        create2Dmatrix<unsigned char>(height, width);
    //  TODO: Check if we want to use histograms
    //  TODO: Add connexe composent for output image

    thresholded_image = compute_otsu_threshold(image, width, height);

    return thresholded_image;
}
