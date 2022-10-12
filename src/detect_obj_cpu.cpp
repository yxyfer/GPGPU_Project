#include "detect_obj.hpp"
#include "utils.hpp"
#include "helpers_images.hpp"
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


unsigned char **detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels) {
    std::string file_save_gray_ref = "../images/gray_scale_ref.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj.jpg";
    
    std::string file_save_blurred_ref = "../images/blurred_ref.jpg";
    std::string file_save_blurred_obj = "../images/blurred_obj.jpg";
    
    std::string diff_image = "../images/diff.jpg";
    
    // Gray Scale 
    unsigned char **gray_ref = to_gray_scale(buffer_ref, width, height, channels);
    unsigned char **gray_obj = to_gray_scale(buffer_obj, width, height, channels);

    save_image(gray_ref, width, height, file_save_gray_ref);
    save_image(gray_obj, width, height, file_save_gray_obj);

    // Blurring
    unsigned char kernel_size = 5;
    
    unsigned char **blurred_ref = apply_blurring(gray_ref, width, height, kernel_size); 
    unsigned char **blurred_obj = apply_blurring(gray_obj, width, height, kernel_size); 
    
    save_image(blurred_ref, width, height, file_save_blurred_ref);
    save_image(blurred_obj, width, height, file_save_blurred_obj);

    //Difference
    unsigned char **diff = difference(blurred_ref, blurred_obj, width, height);
    
    save_image(diff, width, height, diff_image);

    free2Dmatrix(height, gray_ref);
    free2Dmatrix(height, gray_obj);
    free2Dmatrix(height, blurred_ref);
    free2Dmatrix(height, blurred_obj);

    return diff;
}
