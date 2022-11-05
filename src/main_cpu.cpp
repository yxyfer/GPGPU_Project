#include <cstddef>
#include <iostream>
#include <memory>

#include "detect_obj.hpp"
#include "helpers_images.hpp"
#include "utils.hpp"


// Usage: ./main_cpu
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "Mode: CPU\n";
    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    int width, height, channels;
    unsigned char **images = get_images(argc, argv, &width, &height, &channels);

    unsigned char** output = detect_cpu(images[0], images[1], width, height, channels);

    free_images(images, argc - 1);
}
