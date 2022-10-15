#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <limits>

#include "utils.hpp"

#define INFINITY (__builtin_inff())
float inf = std::numeric_limits<float>::infinity();

float var(unsigned int* array, int nb_pixels)
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
    unsigned int* white_pixels =
        (unsigned int*)malloc(nb_whitep * sizeof(unsigned int));
    unsigned int* black_pixels =
        (unsigned int*)malloc((nb_pixels - nb_whitep) * sizeof(unsigned int));

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

    unsigned int var_white = var(white_pixels, nb_whitep);
    unsigned int var_black = var(black_pixels, nb_pixels - nb_whitep);

    // Finish calc

    float weight_whitep = (float)nb_whitep / (float)nb_pixels;
    float weight_blackp = 1 - weight_whitep;

    printf("\n\ncurr threshold %d\n", threshold);
    printf("\nnb white p %d", nb_whitep);
    printf("\nnb pixels: %d", nb_pixels);
    printf("\nvar white %d", var_white);
    printf("\nwheight blackp %f", weight_blackp);
    printf("\nweight white p %f", weight_whitep);

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
    for (int i = 0; i < size; ++i) {
        printf("%f ", array[i]);
        if (array[i] < array[min])
            min = i;
    }
    printf("\nmin %f", array[min]);
    return min;
}

unsigned char** apply_thresholding(unsigned char** image,
                                   unsigned char threshold,
                                   int width,
                                   int height)
{
    unsigned char** first_thresholded_image =
        create2Dmatrix<unsigned char>(height, width);
    unsigned char** second_thresholded_image =
        create2Dmatrix<unsigned char>(height, width);
    // TODO: Add second threshold pass
    // TODO: Check if we want to use histograms

    // Threshold Pass
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (image[x][y] >= 15)
                first_thresholded_image[x][y] = image[x][y];
            else
                first_thresholded_image[x][y] = 0;
        }

    // TODO: MAKE DESCENT!!!!
    // Getting max values in image
    unsigned char max_pixel_vals[height + 1];
    for (int x = 0; x < height; ++x)
        max_pixel_vals[x] = get_max(first_thresholded_image[x], height + 1);
    int max_pixel_val = get_max(max_pixel_vals, height + 1);

    unsigned char test_thresholds[max_pixel_val + 1];
    for (int i = 1; i <= max_pixel_val + 1; ++i)
        test_thresholds[i - 1] = i;

    float threshold_weights[max_pixel_val + 1];
    for (int i = 0; i < max_pixel_val + 1; ++i) {
        threshold_weights[i] = otsu_criteria(first_thresholded_image,
                                             test_thresholds[i], width, height);
        printf("\ncurr Otsu %f\n", threshold_weights[i]);
    }

    int threshold_index = get_min_index(threshold_weights, height);
    printf("\nmin index %d ", threshold_index);
    threshold = test_thresholds[threshold_index];
    printf("\nthreshold %d ", threshold);

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (first_thresholded_image[x][y] >= threshold)
                second_thresholded_image[x][y] = 255;
            else
                second_thresholded_image[x][y] = 0;
        }
    return second_thresholded_image;
}
