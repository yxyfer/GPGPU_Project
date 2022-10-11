#include <cstddef>
#include <memory>
#include <iostream>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "detect_obj.hpp"
#include "utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/* stbi_write_png("sky.png", width, height, channels, img, width * channels); */
/* stbi_write_jpg("sky2.jpg", width, height, channels, img, 100); */
/* stbi_image_free(img); */

// width = col
// height = row

void save_matrix(unsigned char **buffer, int width, int height, std::string filename) {
    unsigned char *sa = (unsigned char *) std::malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            sa[i * width + j] = buffer[i][j];

    char *file = const_cast<char *>(filename.c_str());
    stbi_write_jpg(file, width, height, 1, sa, 100);
}

// Usage: ./mandel
int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    std::string mode = "CPU";

    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    std::string file_save_gray_ref = "../images/gray_scale_ref.jpg";
    std::string file_save_gray_obj = "../images/gray_scale_obj.jpg";
    std::string diff_image = "../images/diff.jpg";
    std::string file_save_blurred_ref = "../images/blurred_ref.jpg";
    std::string file_save_blurred_obj = "../images/blurred_obj.jpg";
    std::string file_save_closing = "../images/closing.jpg";
    std::string file_save_opening = "../images/opening.jpg";


    int width, height, channels;
    unsigned char *ref_image = stbi_load(argv[1], &width, &height, &channels, 0);
    unsigned char *obj_image = stbi_load(argv[2], &width, &height, &channels, 0);

    std::cout << "Reference image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";


    // Gray Scale 
    unsigned char **gray_ref = to_gray_scale(ref_image, width, height, channels);
    unsigned char **gray_obj = to_gray_scale(obj_image, width, height, channels);

    save_matrix(gray_ref, width, height, file_save_gray_ref);
    save_matrix(gray_obj, width, height, file_save_gray_obj);

    // Blurring
    unsigned char kernel_size = 5;
    
    unsigned char **blurred_ref = apply_blurring(gray_ref, width, height, kernel_size); 
    unsigned char **blurred_obj = apply_blurring(gray_obj, width, height, kernel_size); 
    
    save_matrix(blurred_ref, width, height, file_save_blurred_ref);
    save_matrix(blurred_obj, width, height, file_save_blurred_obj);

    //Difference
    unsigned char **diff = difference(blurred_ref, blurred_obj, width, height);

    save_matrix(diff, width, height, diff_image);


    // Opening/Closing
    int es_size = 5;    // Has to be odd
    int es_size2 = 31;  // Has to be odd
    unsigned char **k2 = create_array2D<unsigned char>(es_size, es_size, 1);
    unsigned char **k3 = create_array2D<unsigned char>(es_size2, es_size2, 1);
    /* unsigned char k2_t[7][7] = { { 0, 0, 1, 1, 1, 0, 0 }, */
    /*                              { 0, 1, 1, 1, 1, 1, 0 }, */
    /*                              { 1, 1, 1, 1, 1, 1, 1 }, */
    /*                              { 1, 1, 1, 1, 1, 1, 1 }, */
    /*                              { 1, 1, 1, 1, 1, 1, 1 }, */
    /*                              { 0, 1, 1, 1, 1, 1, 0 }, */
    /*                              { 0, 0, 1, 1, 1, 0, 0 } }; */
    /* for (size_t i = 0; i < 7; i++) */
    /* { */
    /*     for (size_t j = 0; j < 7; j++) */
    /*     { */
    /*         k2[i][j] = k2_t[i][j]; */
    /*     } */
    /* } */


    // Perform closing
    auto output = perform_erosion(diff, k2, height, width, es_size, es_size);
    output = perform_dilation(output, k2, height, width, es_size, es_size);
    
    save_matrix(output, width, height, file_save_closing);

    // Perform opening
    output = perform_dilation(output, k3, height, width, es_size2, es_size2);
    output = perform_erosion(diff, k3, height, width, es_size2, es_size2);
    
    save_matrix(output, width, height, file_save_opening);

    // TODO Free gray_ref, gray_obj, diff
    stbi_image_free(ref_image);
    stbi_image_free(obj_image);
}

