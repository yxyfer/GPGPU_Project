#include "detect_obj.hpp"
#include <iostream>


bool is_number_in_line(struct ImageMat* image, int nb, int line) {
    for (int i = 0; i < image->width; i++)
        if (image->pixel[line][i] == nb)
            return true;

    return false;
}

bool is_number_in_col(struct ImageMat* image, int nb, int col) {
    for (int i = 0; i < image->height; i++)
        if (image->pixel[i][col] == nb)
            return true;

    return false;
}

struct Bbox* box(struct ImageMat *image, int nb) {
    int min_y;
    int min_x;
    int max_y;
    int max_x;

    int i = 0;
    while (i < image->height && !is_number_in_line(image, nb, i))
        i++;

    min_y = i;

    while(i < image->height && is_number_in_line(image, nb, i))
        i++;

    max_y = i;

    i = 0;
    while (i < image->width && !is_number_in_col(image, nb, i))
        i++;

    min_x = i;

    while(i < image->width && is_number_in_col(image, nb, i))
        i++;

    max_x = i;

    struct Bbox* bbox = (struct Bbox*) std::malloc(sizeof(struct Bbox));
    bbox->x = min_x;
    bbox->y = min_y;
    bbox->height = max_y - min_y;
    bbox->width = max_x - min_x;

    return bbox;
}

struct Bbox** get_bbox(struct ImageMat *image, int nb_compo) {
    if (nb_compo <= 0)
        return NULL;

    struct Bbox** bboxes = (struct Bbox**) std::malloc(nb_compo * sizeof(struct Bbox *));
    for (int i = 1; i <= nb_compo; i++) {
        bboxes[i - 1] = box(image, i);
    }

    return bboxes;
}
