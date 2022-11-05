#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <limits>

#include "utils.hpp"
#include "detect_obj.hpp"

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

float otsu_criteria(struct ImageMat* image,
                    unsigned char threshold)
{
    unsigned char** thresholded_image =
        create2Dmatrix<unsigned char>(image->height, image->width);

    unsigned int nb_pixels = image->width * image->height;
    unsigned int nb_whitep = 0;

    for (int x = 0; x < image->height; ++x)
        for (int y = 0; y < image->width; ++y) {
            if (image->pixel[x][y] >= threshold) {
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

    for (int x = 0; x < image->height; ++x)
        for (int y = 0; y < image->width; ++y) {
            if (thresholded_image[x][y] == 1) {
                white_pixels[wp_i] = image->pixel[x][y];
                wp_i++;
            }
            else {
                black_pixels[bp_i] = image->pixel[x][y];
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
    free2Dmatrix(image->height, thresholded_image);
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

int get_max_pixel_val(struct ImageMat* in_image)
{
    // Getting max values in image
    unsigned char max_pixel_vals[in_image->height];
    for (int x = 0; x < in_image->height; ++x)
        max_pixel_vals[x] = get_max(in_image->pixel[x], in_image->height);
    int max_pixel_val = get_max(max_pixel_vals, in_image->height);

    return max_pixel_val;
}

unsigned char get_otsu_threshold(struct ImageMat* in_image)
{
    // getting max image pixel value
    int max_pixel_val = get_max_pixel_val(in_image);

    int range_size = max_pixel_val + 1;
    unsigned char threshold_range[range_size];
    for (int i = 1; i < range_size; ++i)
        threshold_range[i - 1] = i;

    float otsu_vals[range_size];
    for (int i = 0; i < range_size; ++i) {
        unsigned char curr_threshold = threshold_range[i];
        otsu_vals[i] = otsu_criteria(in_image, curr_threshold);
    }

    int threshold_index = get_min_index(otsu_vals, range_size);
    unsigned char otsu_threshold = threshold_range[threshold_index];

    return otsu_threshold;
}

void apply_base_threshold(struct ImageMat* image,
                          unsigned char threshold)
{
    for (int x = 0; x < image->height; ++x)
        for (int y = 0; y < image->width; ++y) {
            if (image->pixel[x][y] >= threshold)
                continue;
            else
                image->pixel[x][y] = 0;
        }
}

void apply_bin_threshold(struct ImageMat* in_image,
                         struct ImageMat* out_image,
                         unsigned char threshold)
{
    for (int x = 0; x < in_image->height; ++x)
        for (int y = 0; y < in_image->width; ++y) {
            if (in_image->pixel[x][y] >= threshold)
                out_image->pixel[x][y] = 255;
            else
                out_image->pixel[x][y] = 0;
        }
}

void compute_otsu_threshold(struct ImageMat* in_image,
                            struct ImageMat* out_image_2)
{
    // TODO: Get two images for the connexe components
    // TODO: But that's after we do a full initial cleanup!

    unsigned char otsu_threshold = get_otsu_threshold(in_image);
    unsigned char otsu_threshold2 = otsu_threshold * 2.5;

    // First threshold saved to out_image_1
    apply_base_threshold(in_image, otsu_threshold - 10);

    // Second threshold saved to out_image_2
    apply_bin_threshold(in_image, out_image_2, otsu_threshold2);

    /* printf("otsu threshold 1 = %i; threshold 2 = %i\n", otsu_threshold, */
    /*        otsu_threshold2); */
}

unsigned char compute_otsu_threshold2(struct ImageMat* in_image, struct ImageMat* temp)
{
    // TODO: Get two images for the connexe components
    // TODO: But that's after we do a full initial cleanup!

    unsigned char otsu_threshold = get_otsu_threshold(in_image);
    unsigned char otsu_threshold2 = otsu_threshold * 2.5;

    // First threshold saved to out_image_1
    apply_base_threshold(in_image, otsu_threshold - 10);

    // Second threshold saved to out_image_2
    apply_bin_threshold(in_image, temp, otsu_threshold2);

    return otsu_threshold2;

    /* printf("otsu threshold 1 = %i; threshold 2 = %i\n", otsu_threshold, */
    /*        otsu_threshold2); */
}

char check_neighbours(struct ImageMat* in_otsu_2,
                      struct ImageMat* in_otsu_1,
                      int x,
                      int y)
{
    unsigned char final_val = in_otsu_2->pixel[x][y];

    if (in_otsu_1->pixel[x - 1][y] > final_val && in_otsu_2->pixel[x - 1][y] != 0)
        // final_val = in_otsu_1[x - 1][y];
        final_val = 255;
    if (in_otsu_1->pixel[x + 1][y] > final_val && in_otsu_2->pixel[x + 1][y] != 0)
        // final_val = in_otsu_1[x + 1][y];
        final_val = 255;
    if (in_otsu_1->pixel[x][y - 1] > final_val && in_otsu_2->pixel[x][y - 1] != 0)
        // final_val = in_otsu_1[x][y - 1];
        final_val = 255;
    if (in_otsu_1->pixel[x][y + 1] > final_val && in_otsu_2->pixel[x][y + 1] != 0)
        // final_val = in_otsu_1[x][y + 1];
        final_val = 255;

    // TODO: CHECK IF WE NEED DIAGONALS?

    char changed = 0;
    if (final_val != in_otsu_2->pixel[x][y])
        changed = 1;

    in_otsu_2->pixel[x][y] = final_val;
    return changed;
}

// TODO: TAKE CARE OF CORNERS
char propagate(struct ImageMat* in_otsu_2,
               struct ImageMat* in_otsu_1)
{
    char changed = 0;

    for (int i = 1; i < in_otsu_2->height - 1; ++i)
        for (int j = 1; j < in_otsu_2->width - 1; ++j) {
            char has_changed = check_neighbours(in_otsu_2, in_otsu_1, i, j);
            if (has_changed)
                changed = 1;
        }

    return changed;
}

/* void connexe_components(struct ImageMat* in_otsu_1, */
/*                         struct ImageMat* in_otsu_2) */
/* { */
/*     char l_changed = 1; */
/*     while (l_changed) { */
/*         l_changed = propagate(in_otsu_2, in_otsu_1); */
/*     } */
/* } */

void components(struct ImageMat* img, struct ImageMat* temp, int x, int y, int val) {
    if (x < 0 || x >= img->height || y < 0 || y >= img->width || img->pixel[x][y] % 255 == 0)
        return;

    temp->pixel[x][y] = val;
    img->pixel[x][y] = 255;
    components(img, temp, x + 1, y, val);
    components(img, temp, x - 1, y, val);
    components(img, temp, x, y + 1, val);
    components(img, temp, x, y - 1, val);
}

int connexe_components(struct ImageMat *img_1, struct ImageMat* temp) {
    int number = 20;
    for (int i = 0; i < img_1->height; i++) {
        for (int j = 0; j < img_1->width; j++) {
            if (temp->pixel[i][j] == 255) {
                img_1->pixel[i][j] = 1;
                components(img_1, temp, i, j, number);
                number++;
            }
        }
    }

    return number - 1;
}

void compute_threshold(struct ImageMat* base_image,
                       struct ImageMat* temp_image)
{
    /* compute_otsu_threshold(base_image, temp_image); */
    unsigned char th = compute_otsu_threshold2(base_image, temp_image);
    std::cout << (int)th << '\n';
    int nb_compo = connexe_components(base_image, temp_image);
    std::cout << (int)nb_compo << '\n';
    /* connexe_components(base_image, temp_image); */

    swap_matrix(base_image, temp_image);
}
