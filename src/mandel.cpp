#include <cstddef>
#include <memory>
#include <iostream>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "detect_obj.hpp"
#include "utils.hpp"
#include "helpers_images.hpp"


// width = col
// height = row


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

    std::string file_save_closing = "../images/closing.jpg";
    std::string file_save_opening = "../images/opening.jpg";
    std::string file_save_threshold_base = "../images/threshold_base.jpg";

    int width, height, channels;
    unsigned char *ref_image = load_image(argv[1], &width, &height, &channels);
    unsigned char *obj_image = load_image(argv[2], &width, &height, &channels);

    std::cout << "Reference image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";

    // Get difference
    unsigned char **diff = detect_cpu(ref_image, obj_image, width, height, channels);

    // Opening/Closing
    int es_size = 5;    // Has to be odd
    int es_size2 = 11;  // Has to be odd

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
    
    save_image(output, width, height, file_save_closing);

    // Perform opening
    output = perform_dilation(output, k3, height, width, es_size2, es_size2);
    output = perform_erosion(output, k3, height, width, es_size2, es_size2);
    
    save_image(output, width, height, file_save_opening);

    // Perform threshold
    //output = apply_thresholding(output, 15, width, height);
    output = apply_thresholding(output, 118, width, height);
    save_image(output, width, height, file_save_threshold_base);

    // TODO: free
    free_image(ref_image);
    free_image(obj_image);
}

