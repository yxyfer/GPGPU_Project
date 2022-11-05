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

    // Freeing variables
    free2Dmatrix(height, thresholded_image);
    free(white_pixels);
    free(black_pixels);

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

    return otsu_threshold;
}

void apply_base_threshold(unsigned char** in_image,
                          unsigned char** out_image,
                          unsigned char threshold,
                          int width,
                          int height)
{
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (in_image[x][y] >= threshold)
                out_image[x][y] = in_image[x][y];
            else
                out_image[x][y] = 0;
        }
}

void apply_bin_threshold(unsigned char** in_image,
                         unsigned char** out_image,
                         unsigned char threshold,
                         int width,
                         int height)
{
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (in_image[x][y] >= threshold)
                out_image[x][y] = 255;
            else
                out_image[x][y] = 0;
        }
}

void compute_otsu_threshold(unsigned char** in_image,
                            unsigned char**& out_image_1,
                            unsigned char**& out_image_2,
                            int width,
                            int height)
{
    // TODO: Get two images for the connexe components
    // TODO: But that's after we do a full initial cleanup!

    unsigned char otsu_threshold = get_otsu_threshold(in_image, width, height);
    unsigned char otsu_threshold2 = otsu_threshold * 2.5;

    // First threshold saved to out_image_1
    apply_base_threshold(in_image, out_image_1, otsu_threshold - 10, width,
                         height);

    // Second threshold saved to out_image_2
    apply_bin_threshold(out_image_1, out_image_2, otsu_threshold2, width,
                        height);

    printf("otsu threshold 1 = %i; threshold 2 = %i\n", otsu_threshold,
           otsu_threshold2);
}

void init_connexe_components(unsigned char** L,
                             unsigned char** in_otsu_2,
                             int width,
                             int height)
{
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            L[i][j] = in_otsu_2[i][j];
        }
}

char check_neighbours(unsigned char** L,
                      unsigned char** in_otsu_1,
                      int x,
                      int y)
{
    unsigned char final_val = L[x][y];

    if (in_otsu_1[x - 1][y] > final_val && L[x - 1][y] != 0)
        // final_val = in_otsu_1[x - 1][y];
        final_val = 255;
    if (in_otsu_1[x + 1][y] > final_val && L[x + 1][y] != 0)
        // final_val = in_otsu_1[x + 1][y];
        final_val = 255;
    if (in_otsu_1[x][y - 1] > final_val && L[x][y - 1] != 0)
        // final_val = in_otsu_1[x][y - 1];
        final_val = 255;
    if (in_otsu_1[x][y + 1] > final_val && L[x][y + 1] != 0)
        // final_val = in_otsu_1[x][y + 1];
        final_val = 255;

    // TODO: CHECK IF WE NEED DIAGONALS?

    char changed = 0;
    if (final_val != L[x][y])
        changed = 1;

    L[x][y] = final_val;
    return changed;
}

// TODO: TAKE CARE OF CORNERS
char propagate(unsigned char**& L,
               unsigned char** in_otsu_1,
               int width,
               int height)
{
    char changed = 0;

    for (int i = 1; i < height - 1; ++i)
        for (int j = 1; j < width - 1; ++j) {
            char has_changed = check_neighbours(L, in_otsu_1, i, j);
            if (has_changed)
                changed = 1;
        }

    return changed;
}

unsigned char** connexe_components(unsigned char** in_otsu_1,
                                   unsigned char** in_otsu_2,
                                   int width,
                                   int height)
{
    unsigned char** L = create2Dmatrix<unsigned char>(height, width);

    init_connexe_components(L, in_otsu_2, width, height);

    char l_changed = 1;
    while (l_changed) {
        l_changed = propagate(L, in_otsu_1, width, height);
    }

    return L;
}

unsigned char** compute_threshold(unsigned char** image, int width, int height)
{
    //  TODO: Add connexe composent for output image

    unsigned char** otsu_threshold1 =
        create2Dmatrix<unsigned char>(height, width);

    // TODO: Change this to the actual connexe component linker
    unsigned char** thresholded_image =
        create2Dmatrix<unsigned char>(height, width);

    compute_otsu_threshold(image, otsu_threshold1, thresholded_image, width,
                           height);

    unsigned char** connexe_component =
        connexe_components(otsu_threshold1, thresholded_image, width, height);

    // Free
    free2Dmatrix(height, otsu_threshold1);
    free2Dmatrix(height, thresholded_image);

    // return thresholded_image;
    return connexe_component;
}
