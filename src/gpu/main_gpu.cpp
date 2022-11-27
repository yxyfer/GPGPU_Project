#include <cstddef>
#include <iostream>
#include <memory>

#include "detect_obj_gpu.hpp"
#include "helpers_images.hpp"

// Usage: ./main_gpu
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    if (argc < 3)
    {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    int width, height, channels;
    unsigned char **images = get_images(argc, argv, &width, &height, &channels);

    main_detection_gpu(images, argc - 1, width, height, channels);


    free_images(images, argc - 1);
}
