#include <cstddef>
#include <iostream>
#include <memory>

#include "detect_obj.hpp"
#include "helpers_images.hpp"
#include "utils.hpp"

void format_bbox(struct Bbox**bbox, int compo,  char* name, bool last) {
    std::cout << "  " << '"'  << name << '"' << ": [\n";
    for (int i = 0; i < compo; i++) {
        std::cout << "    "  << '[' << bbox[i]->x << ", " << bbox[i]->y << ", " << bbox[i]->width << ", " << bbox[i]->height << "]";
        if (i + 1 != compo) {
            std::cout << ',';
        }
        std::cout <<'\n';
    }

    if (!last)
        std::cout << "  ],\n";
    else
        std::cout << "  ]\n";
}

// Usage: ./main_cpu
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    /* std::cout << "Mode: CPU\n"; */
    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    int width, height, channels, nb_obj;
    unsigned char **images = get_images(argc, argv, &width, &height, &channels);

    struct Bbox** bboxes = detect_cpu(images[0], images[1], width, height, channels, &nb_obj);

    std::cout << "{\n";
    format_bbox(bboxes, nb_obj, argv[2], true);
    std::cout << "}\n";

    for (int i = 0; i < nb_obj; i++) {
        std::free(bboxes[i]);
    }

    std::free(bboxes);
    free_images(images, argc - 1);
    
}
