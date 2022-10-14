#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "utils.hpp"
#define INFINITY (__builtin_inff())

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

unsigned char otsu_criteria(unsigned char** image,
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
                thresholded_image[x][y] = 255;
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
            if (image[x][y] >= threshold) {
                white_pixels[wp_i] = image[x][y];
                wp_i++;
            }
            else {
                black_pixels[bp_i] = image[x][y];
                bp_i++;
            }
        }

    nb_whitep = wp_i;

    unsigned char var_white = var(white_pixels, nb_whitep);
    unsigned char var_black = var(black_pixels, nb_pixels - nb_whitep);

    // Finish calc

    printf("\nvar white %d", var_white);
    printf("\nnb white p %d", nb_whitep);
    printf("\n nb pixels: %d", nb_pixels);

    float weight_whitep = (float)nb_whitep / (float)nb_pixels;
    float weight_blackp = 1 - weight_whitep;

    printf("\nthreshold %d", threshold);
    printf("\nwheight blackp %f", weight_blackp);
    printf("\nweight white p %f\n", weight_whitep);

    if (weight_whitep == 0 || weight_blackp == 0)
        return 255;

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

unsigned char get_min_index(float* array, int size)
{
    unsigned char min = 0;
    for (int i = 0; i < size; ++i) {
        min = array[min] > array[i] ? i : min;
    }
    printf("\nmin %d", array[min]);
    return min;
}

unsigned char** apply_thresholding(unsigned char** image,
                                   unsigned char threshold,
                                   int width,
                                   int height)
{
    unsigned char** thresholded_image =
        create2Dmatrix<unsigned char>(height, width);
    // TODO: Add second threshold pass
    // TODO: Check if we want to use histograms

    // TODO: MAKE DESCENT!!!!
    // Getting max values in image
    unsigned char max_pixel_vals[height + 1];
    for (int x = 0; x < height; ++x)
        max_pixel_vals[x] = get_max(image[x], height + 1);
    int max_pixel_val = get_max(max_pixel_vals, height + 1);

    unsigned char test_thresholds[max_pixel_val + 1];
    for (int i = 1; i <= max_pixel_val + 1; ++i)
        test_thresholds[i - 1] = i;

    float threshold_weights[max_pixel_val + 1];
    for (int i = 0; i < max_pixel_val + 1; ++i) {
        threshold_weights[i] =
            otsu_criteria(image, test_thresholds[i], width, height);
        printf("\nOtsu %d ", threshold_weights[i]);
    }

    int threshold_index = get_min_index(threshold_weights, height);
    printf("\nmin index %d ", threshold_index);
    threshold = test_thresholds[threshold_index];
    printf("\nthreshold %d ", threshold);

    // Threshold Pass
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (image[x][y] >= threshold)
                thresholded_image[x][y] = 255;
            else
                thresholded_image[x][y] = 0;
        }

    return thresholded_image;
}
