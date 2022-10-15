#include <CLI/CLI.hpp>
#include <cstddef>
#include <iostream>
#include <memory>
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

void save_matrix(unsigned char **buffer, int width, int height,
                 std::string filename)
{
    unsigned char *sa =
        (unsigned char *)std::malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            sa[i * width + j] = buffer[i][j];

    char *file = const_cast<char *>(filename.c_str());
    stbi_write_jpg(file, width, height, 1, sa, 100);
}

// Usage: ./mandel
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    std::string mode = "CPU";

    if (argc < 3)
    {
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
    unsigned char *ref_image =
        stbi_load(argv[1], &width, &height, &channels, 0);
    unsigned char *obj_image =
        stbi_load(argv[2], &width, &height, &channels, 0);

    std::cout << "Reference image: " << argv[1] << " | " << height << "x"
              << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " << height << "x" << width
              << "x" << channels << "\n";

    // Gray Scale
    unsigned char **gray_ref =
        to_gray_scale(ref_image, width, height, channels);
    unsigned char **gray_obj =
        to_gray_scale(obj_image, width, height, channels);

    save_matrix(gray_ref, width, height, file_save_gray_ref);
    save_matrix(gray_obj, width, height, file_save_gray_obj);

    // Blurring
    // TODO: MAKE KERNEL GENERATOR - with better calculations
    unsigned char kernel_size = 3;
    unsigned char **kernel =
        (unsigned char **)malloc(kernel_size * sizeof(unsigned char *));

    for (int i = 0; i < kernel_size; i++)
        kernel[i] =
            (unsigned char *)malloc(kernel_size * sizeof(unsigned char));

    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++)
            kernel[i][j] = 1 * (i == 1 ? 2 : 1) * (j == 1 ? 2 : 1);

    unsigned char **blurred_ref =
        blurring(gray_ref, kernel, width, height, kernel_size);
    save_matrix(blurred_ref, width, height, file_save_blurred_ref);

    unsigned char **blurred_obj =
        blurring(gray_obj, kernel, width, height, kernel_size);
    save_matrix(blurred_obj, width, height, file_save_blurred_obj);

    unsigned char **diff = difference(blurred_ref, blurred_obj, width, height);
    save_matrix(diff, width, height, diff_image);

    int es_size = 11;
    int es_size2 = 25;
    unsigned char **k1 = circular_kernel(es_size);
    unsigned char **k2 = circular_kernel(es_size2);

    print_mat(k1, es_size, es_size);

    // Perform closing/opening
    auto closing = perform_erosion(diff, k1, height, width, es_size, es_size);
    closing = perform_dilation(closing, k1, height, width, es_size, es_size);

    auto opening =
        perform_dilation(closing, k2, height, width, es_size2, es_size2);
    opening = perform_erosion(opening, k2, height, width, es_size2, es_size2);

    save_matrix(closing, width, height, file_save_closing);

    save_matrix(opening, width, height, file_save_opening);

    // TODO Free gray_ref, gray_obj, diff
    stbi_image_free(ref_image);
    stbi_image_free(obj_image);
}
