#include <cstddef>
#include <memory>
#include <iostream>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "detect.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* stbi_write_png("sky.png", width, height, channels, img, width * channels); */
/* stbi_write_jpg("sky2.jpg", width, height, channels, img, 100); */
/* stbi_image_free(img); */

// width = col
// height = row

void save_matrix(unsigned char **buffer, int width, int height) {
    unsigned char *sa = (unsigned char *) std::malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            sa[i * width + j] = buffer[i][j];

    stbi_write_jpg("../images/gray_scale.jpg", width, height, 1, sa, 100);
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

    int width, height, channels;
    unsigned char *ref_image = stbi_load(argv[1], &width, &height, &channels, 0);
    unsigned char *obj_image = stbi_load(argv[1], &width, &height, &channels, 0);

    std::cout << "Reference image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";
    std::cout << "Object image: " << argv[1] << " | " <<  height << "x" << width << "x" << channels << "\n";

    unsigned char **gray_ref = to_gray_scale(ref_image, width, height, channels);

    save_matrix(gray_ref, width, height);

    // TODO Free gray_ref
    stbi_image_free(ref_image);
    stbi_image_free(obj_image);
}

