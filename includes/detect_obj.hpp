#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

struct ImageMat
{
    unsigned char **pixel;
    int height;
    int width;
};

struct GaussianKernel
{
    double **kernel;
    unsigned char size;
};

struct MorphologicalKernel
{
    unsigned char **kernel;
    int size;
};

struct Bbox
{
    int x;
    int y;
    int height;
    int width;
};

///// FILE: struct_utils.cpp
struct ImageMat *new_matrix(int height, int width);

///// FILE: struct_utils.cpp
void swap_matrix(struct ImageMat *a, struct ImageMat *b);

///// FILE: struct_utils.cpp
struct GaussianKernel *create_gaussian_kernel(unsigned char size);

///// FILE: struct_utils.cpp
struct MorphologicalKernel *circular_kernel(int kernel_size);

/// FILE: struct_utils.cpp
void freeImageMat(struct ImageMat *a);

/// FILE: struct_utils.cpp
void freeGaussianKernel(struct GaussianKernel *kernel);

/// FILE: struct_utils.cpp
void freeMorphologicalKernel(struct MorphologicalKernel *kernel);

// FILE: detect_obj_cpu.cpp
/// \param buffer_ref: The RGBA24 image buffer
/// \param buffer_obj: The RGBA24 image buffer
/// \param width: Image width
/// \param height: Image height
/// \param channels: Image number of channels
/// \param nb_obj: Return the number of objects in the image
struct Bbox **main_detection_test(unsigned char *buffer_ref,
                                  unsigned char *buffer_obj, int width,
                                  int height, int channels, int *nb_obj);

// FILE: detect_obj_cpu.cpp
/// \param images: List of images, ref in the fisrt position
/// \param length: Length of the list
/// \param width: Image width
/// \param height: Image height
/// \param channels: Image number of channels
/// \param nb_objs: Return the number of objects in the images
struct Bbox ***main_detection(unsigned char **images, int length, int width,
                              int height, int channels, int *nb_objs);

// FILE: detect_obj_cpu.cpp
/// \param src: The RGBA24 image buffer
/// \param dst: The struct image to store the information
/// \param width: Image width
/// \param height: Image height
/// \param channels: Image number of channels
void to_gray_scale(unsigned char *src, struct ImageMat *dst, int width,
                   int height, int channels);

// FILE: detect_obj_cpu.cpp
/// Store the result in the struct image obj
/// \param ref: The struct image of the reference
/// \param obj: The struct image of the object
void difference(struct ImageMat *ref, struct ImageMat *obj);

// FILE: blur_cpu.cpp
// Apply gaussian blur to the image
/// \param image: The struct image
/// \param temp: A temp struct image
/// \param kernel: The Gaussian kernel
void apply_blurring(struct ImageMat *image, struct ImageMat *temp,
                    struct GaussianKernel *kernel);

// FILE: opening.cpp
/// \param input: The struct image
/// \param temp: A temp struct image
/// \param kernel: A morphological kernel
void perform_dilation(struct ImageMat *input, struct ImageMat *temp,
                      struct MorphologicalKernel *kernel);

// FILE: opening.cpp
/// \param input: The struct image
/// \param temp: A temp struct image
/// \param kernel: A morphological kernel
void perform_erosion(struct ImageMat *input, struct ImageMat *temp,
                     struct MorphologicalKernel *kernel);

// FILE: threshold_cpu.cpp
/// \param image: The struct image
/// \param temp: A temp struct image
int threshold(struct ImageMat *image, struct ImageMat *temp);

// FILE: bbox.cpp
/// \param image: The struct image
/// \param nb_compo: Number of components
struct Bbox **get_bbox(struct ImageMat *image, int nb_compo);

/// \param buffer_ref The RGBA24 image buffer
/// \param buffer_obj The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param channels Image number of channels
