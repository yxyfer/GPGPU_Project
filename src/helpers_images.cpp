#include "helpers_images.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char *load_image(char *path, int *width, int *height, int *channels) {
    return stbi_load(path, width, height, channels, 0);
}

void save_image(unsigned char **image, int width, int height, std::string filename) {
    unsigned char *flat_image = (unsigned char *) std::malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            flat_image[i * width + j] = image[i][j];

    char *file = const_cast<char *>(filename.c_str());
    stbi_write_jpg(file, width, height, 1, flat_image, 100);

    free(flat_image);
}

void free_image(unsigned char* image) {
    stbi_image_free(image);
}
