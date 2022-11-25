#include "gpu_functions.hpp"
#include "../helpers_images.hpp"
#include <iostream>


int main(int argc, char** argv) {
    int width, height, channels;
    /* int width = 4; */
    /* int height = 4; */

    /* unsigned char **images = (unsigned char **)std::malloc((height * width) * sizeof(unsigned char *)); */

    /* for (int i = 0; i < height; i++) { */
    /*     images[i] = (unsigned char *)std::malloc(width * sizeof(unsigned char)); */
    /*     for (int j = 0; j < width; j++) { */
    /*         images[i][j] = 1; */
    /*     } */
    /* } */
    
    /* unsigned char images[] = {1, 1, 1, 12, 1, 12, 1, 1, 11, 1, 9, 1, 1, 1, 1, 1}; */
    /* unsigned char images[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; */
    unsigned char* images = load_image(argv[1], &width, &height, &channels);

    /* std::cout << argv[1] << " | " << width << "x" << height << "x" << channels << '\n'; */

    test_function(images, (size_t) height, (size_t) width);
    return 0;
}
