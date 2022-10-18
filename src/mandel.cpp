#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <cstddef>
#include <iostream>
#include <memory>

#include "detect_obj.hpp"
#include "helpers_images.hpp"
#include "utils.hpp"

// Usage: ./mandel
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::string mode = "CPU";

    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    std::string file_save_closing = "../images/closing.jpg";
    std::string file_save_opening = "../images/opening.jpg";
    std::string file_save_threshold_base = "../images/threshold_base.jpg";

    int width, height, channels;
    unsigned char* ref_image = load_image(argv[1], &width, &height, &channels);
    unsigned char* obj_image = load_image(argv[2], &width, &height, &channels);

    std::cout << "Reference image: " << argv[1] << " | " << height << "x"
              << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " << height << "x" << width
              << "x" << channels << "\n";

    detect_gpu(ref_image, obj_image, width, height, channels);

    /* // Get difference */
    /* unsigned char** diff = */
    /*     detect_cpu(ref_image, obj_image, width, height, channels); */

    /* int es_size = 5; */
    /* int es_size2 = 11; */
    /* unsigned char** k1 = circular_kernel(es_size); */
    /* unsigned char** k2 = circular_kernel(es_size2); */

    /* print_mat(k1, es_size, es_size); */

    /* // Perform closing/opening */
    /* auto closing = perform_erosion(diff, k1, height, width, es_size, es_size); */
    /* closing = perform_dilation(closing, k1, height, width, es_size, es_size); */

    /* save_image(closing, width, height, file_save_closing); */

    /* // Perform opening */
    /* auto opening = */
    /*     perform_dilation(closing, k2, height, width, es_size2, es_size2); */
    /* opening = perform_erosion(opening, k2, height, width, es_size2, es_size2); */

    /* save_image(opening, width, height, file_save_opening); */

    /* // Perform threshold */
    /* // output = apply_thresholding(output, 15, width, height); */
    /* auto thresh_img = compute_threshold(opening, width, height); */
    /* save_image(thresh_img, width, height, file_save_threshold_base); */

    // TODO: free
    free_image(ref_image);
    free_image(obj_image);
}
