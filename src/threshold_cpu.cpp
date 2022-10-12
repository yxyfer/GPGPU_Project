#include "utils.hpp"
#include <math.h>

#  define INFINITY (__builtin_inff ())

float var(unsigned char *array, int nb_pixels) {
    float sum = 0.0, mean, variance = 0.0;

    for(int i = 0; i < nb_pixels; ++i)
        sum += array[i];

    mean = sum / nb_pixels;

    // TODO: NOT OPTIMISED WHAT SO EVER!
    for(int i = 0; i < nb_pixels; ++i)
        variance += pow(array[i] - mean, 2);

    variance /= nb_pixels;

    return variance;
}

unsigned char otsu_criteria(unsigned char **image, unsigned char threshold, int width, int height) {
    
    unsigned char** thresholded_image = create2Dmatrix<unsigned char>(height, width);

    unsigned int nb_pixels = width * height;
    unsigned int nb_whitep = 0;


    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (image[x][y] >= threshold) {
                thresholded_image[x][y] = 255;
                nb_whitep++;
            } else
                thresholded_image[x][y] = 0;
        }

    // Seperate white and black pixels into two arrays
    unsigned char *white_pixels = (unsigned char *) malloc(nb_whitep * sizeof(unsigned char));
    unsigned char *black_pixels = (unsigned char *) malloc((nb_pixels - nb_whitep) * sizeof(unsigned char));

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

    unsigned char var_white = var(white_pixels, nb_whitep);
    unsigned char var_black = var(black_pixels, nb_pixels - nb_whitep);

    // Finish calc
    float weight_whitep = nb_whitep / nb_pixels;
    float weight_blackp = 1 - weight_whitep;

    if (weight_whitep == 0 || weight_blackp == 0)
        return INFINITY;


    return weight_whitep * var_white + weight_blackp * var_black;
}

unsigned char **apply_thresholding(unsigned char** image, unsigned char threshold, int width, int height) {
    unsigned char** thresholded_image = create2Dmatrix<unsigned char>(height, width);

    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y) {
            if (image[x][y] >= threshold)
                thresholded_image[x][y] = 255;
            else
                thresholded_image[x][y] = 0;
        }

    return thresholded_image;
}
