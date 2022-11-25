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

void display_result(struct Bbox*** boxes, int *nb_objs, int argc, char** argv) {
    std::cout << "{\n";
    for (int i = 0; i < argc - 2; i++) {
        format_bbox(boxes[i], nb_objs[i], argv[i + 2], i == (argc - 3));
    }
    std::cout << "}\n";
}

void free_boxes(struct Bbox*** boxes, int length, int* nb_objs) {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < nb_objs[i]; j++) {
            std::free(boxes[i][j]);
        }
        std::free(boxes[i]);
    }

    std::free(boxes);
}

// Usage: ./main_cpu
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    if (argc < 3) {
        std::cout << "Minimum two images are needed\n";
        return 1;
    }

    int width, height, channels;
    unsigned char **images = get_images(argc, argv, &width, &height, &channels);

    int *nb_objs = (int *) std::malloc((argc - 2) * sizeof(int));

    struct Bbox*** all_boxes = main_detection(images, argc - 1, width, height, channels, nb_objs);
    display_result(all_boxes, nb_objs, argc, argv);

    free_boxes(all_boxes, argc - 2, nb_objs);
    std::free(nb_objs);
    free_images(images, argc - 1);
}
