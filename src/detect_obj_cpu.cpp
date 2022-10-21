#include "detect_obj.hpp"
#include "utils.hpp"
#include "helpers_images.hpp"
#include <iostream>


void swap_matrix(unsigned char ***a, unsigned char ***b) {
    unsigned char **temp = *a;
    *a = *b;
    *b = temp;
}

// Luminosity Method: gray scale -> 0.3 * R + 0.59 * G + 0.11 * B;
void to_gray_scale(unsigned char *src, unsigned char **dst, int width, int height, int channels) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            dst[r][c] = (0.30 * src[(r * width + c) * channels] +       // R
                         0.59 * src[(r * width + c) * channels + 1] +   // G
                         0.11 * src[(r * width + c) * channels + 2]);   // B
        }
    }
}

// Perform |gray_ref - gray_obj|
void difference(unsigned char **gray_ref, unsigned char **gray_obj, int width, int height) {
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++)
            gray_obj[r][c] = abs(gray_ref[r][c] - gray_obj[r][c]);
}


unsigned char **detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj, int width, int height, int channels) {
    std::string file_save_gray_ref = "../images/gray_scale_ref.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj.jpg";
    
    std::string file_save_blurred_ref = "../images/blurred_ref.jpg";
    std::string file_save_blurred_obj = "../images/blurred_obj.jpg";
    
    std::string diff_image = "../images/diff.jpg";
    
    std::string file_save_closing = "../images/closing.jpg";
    std::string file_save_opening = "../images/opening.jpg";

    // Create 2D ref and obj matrix and 2D temp matrix
    unsigned char **ref_matrix = create2Dmatrix<unsigned char>(height, width);
    unsigned char **obj_matrix = create2Dmatrix<unsigned char>(height, width);
    unsigned char **temp_matrix = create2Dmatrix<unsigned char>(height, width);
    
    // Gray Scale 
    to_gray_scale(buffer_ref, ref_matrix, width, height, channels);
    to_gray_scale(buffer_obj, obj_matrix, width, height, channels);

    save_image(ref_matrix, width, height, file_save_gray_ref);
    save_image(obj_matrix, width, height, file_save_gray_obj);

    // Blurring
    unsigned char kernel_size = 5;
    double **kernel = create_gaussian_kernel(kernel_size);
    
    apply_blurring(ref_matrix, temp_matrix, width, height, kernel, kernel_size); 
    swap_matrix(&temp_matrix, &ref_matrix);
    apply_blurring(obj_matrix, temp_matrix, width, height, kernel, kernel_size);
    swap_matrix(&temp_matrix, &obj_matrix);

    free2Dmatrix(kernel_size, kernel);
    
    save_image(ref_matrix, width, height, file_save_blurred_ref);
    save_image(obj_matrix, width, height, file_save_blurred_obj);

    //Difference change obj
    difference(ref_matrix, obj_matrix, width, height);
    
    save_image(obj_matrix, width, height, diff_image);

    // Perform closing/opening
    int es_size = 5;
    int es_size2 = 11;
    unsigned char** k1 = circular_kernel(es_size);
    unsigned char** k2 = circular_kernel(es_size2);

    // Perform closing
    auto closing = perform_closing(obj_matrix, k1, height, width, es_size);
    save_image(closing, width, height, file_save_closing);

    // Perform opening
    auto opening = perform_opening(closing, k2, height, width, es_size2);
    save_image(opening, width, height, file_save_opening);

    free2Dmatrix(height, ref_matrix);
    free2Dmatrix(height, obj_matrix);
    free2Dmatrix(height, closing);
    free2Dmatrix(es_size, k1);
    free2Dmatrix(es_size2, k2);

    return opening;
}
