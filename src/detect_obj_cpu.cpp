#include "detect_obj.hpp"
#include "utils.hpp"
#include <iostream>


// Luminosity Method: gray scale -> 0.3 * R + 0.59 * G + 0.11 * B;
unsigned char **to_gray_scale(unsigned char *buffer, int width, int height, int channels) {
    unsigned char **gray_scale = create2Dmatrix<unsigned char>(height, width);
    
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            gray_scale[r][c] = (0.30 * buffer[(r * width + c) * channels] +       // R
                                0.59 * buffer[(r * width + c) * channels + 1] +   // G
                                0.11 * buffer[(r * width + c) * channels + 2]);   // B
        }
    }

    return gray_scale;
}

// Perform |gray_ref - gray_obj|
unsigned char **difference(unsigned char **gray_ref, unsigned char **gray_obj, int width, int height) {
    unsigned char **diff = create2Dmatrix<unsigned char>(height, width);

    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++)
            diff[r][c] = abs(gray_ref[r][c] - gray_obj[r][c]);

    return diff;
}


void detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels) {
    return;
}
