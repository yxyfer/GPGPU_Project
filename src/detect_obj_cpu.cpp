#include "detect_obj.hpp"
#include "utils.hpp"
#include "helpers_images.hpp"
#include <iostream>


// Luminosity Method: gray scale -> 0.3 * R + 0.59 * G + 0.11 * B;
void to_gray_scale(unsigned char *src, struct ImageMat *dst, int width, int height, int channels) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            dst->pixel[r][c] = (0.30 * src[(r * width + c) * channels] +       // R
                                0.59 * src[(r * width + c) * channels + 1] +   // G
                                0.11 * src[(r * width + c) * channels + 2]);   // B
        }
    }
}

// Perform |gray_ref - gray_obj|
void difference(struct ImageMat *ref, struct ImageMat* obj) {
    for (int r = 0; r < ref->height; r++)
        for (int c = 0; c < ref->width; c++)
            obj->pixel[r][c] = abs(ref->pixel[r][c] - obj->pixel[r][c]);
}

struct Bbox **object_detection(struct ImageMat *blur_ref,
                               unsigned char *buffer_obj,
                               struct ImageMat *obj,
                               struct ImageMat *temp,
                               struct GaussianKernel *g_kernel,
                               struct MorphologicalKernel *k1,
                               struct MorphologicalKernel *k2,
                               int *nb_obj,
                               int channels) {
    
    to_gray_scale(buffer_obj, obj, blur_ref->width, blur_ref->height, channels);
    
    apply_blurring(obj, temp, g_kernel);
    difference(blur_ref, obj);
    
    // Perform closing
    perform_erosion(obj, temp, k1);
    perform_dilation(obj, temp, k1);
        
    // Perform opening
    perform_dilation(obj, temp, k2);
    perform_erosion(obj, temp, k2);

    int nb_components = compute_threshold(obj, temp);

    struct Bbox** bboxes = get_bbox(obj, nb_components);
    *nb_obj = nb_components;

    return bboxes;
}

struct Bbox ***main_detection(unsigned char **images, int length, int width, int height, int channels, int *nb_objs) {
    struct ImageMat *ref_image = new_matrix(height, width);
    struct ImageMat *obj_image = new_matrix(height, width);
    struct ImageMat *temp_image = new_matrix(height, width);

    to_gray_scale(images[0], ref_image, width, height, channels);

    struct GaussianKernel* g_kernel = create_gaussian_kernel(5);
    apply_blurring(ref_image, temp_image, g_kernel);
    
    struct MorphologicalKernel* k1 = circular_kernel(5);
    struct MorphologicalKernel* k2 = circular_kernel(11);

    struct Bbox*** all_boxes = (struct Bbox ***) std::malloc((length - 1) * sizeof(struct BBox **));
    int nb_obj;

    for (int i = 1; i < length; i++) {
        all_boxes[i - 1] = object_detection(ref_image, images[i], obj_image, temp_image, g_kernel, k1, k2, &nb_obj, channels);
        nb_objs[i - 1] = nb_obj;
    }

    freeImageMat(ref_image);
    freeImageMat(temp_image);
    freeImageMat(obj_image);
    freeGaussianKernel(g_kernel);
    freeMorphologicalKernel(k1);
    freeMorphologicalKernel(k2);
    
    return all_boxes;
}

struct Bbox **detect_cpu(unsigned char *buffer_ref, unsigned char *buffer_obj,
                           int width, int height, int channels, int *nb_obj) {
    /* std::string file_save_gray_ref = "../images/gray_scale_ref.jpg"; */
    /* std::string file_save_gray_obj = "../images/gray_scale_obj.jpg"; */
    
    /* std::string file_save_blurred_ref = "../images/blurred_ref.jpg"; */
    /* std::string file_save_blurred_obj = "../images/blurred_obj.jpg"; */
    
    /* std::string diff_image = "../images/diff.jpg"; */
    
    /* std::string file_save_closing = "../images/closing.jpg"; */
    /* std::string file_save_opening = "../images/opening.jpg"; */
    
    /* std::string file_save_threshold_base = "../images/threshold_base.jpg"; */

    // Create 2D ref and obj matrix and 2D temp matrix

    struct ImageMat *ref_image = new_matrix(height, width);
    struct ImageMat *obj_image = new_matrix(height, width);
    struct ImageMat *temp_image = new_matrix(height, width);

    // Gray Scale 
    to_gray_scale(buffer_ref, ref_image, width, height, channels);
    to_gray_scale(buffer_obj, obj_image, width, height, channels);

    /* save_image(ref_image->pixel, width, height, file_save_gray_ref); */
    /* save_image(obj_image->pixel, width, height, file_save_gray_obj); */

    // Blurring
    struct GaussianKernel* g_kernel = create_gaussian_kernel(5);
    
    apply_blurring(ref_image, temp_image, g_kernel);
    apply_blurring(obj_image, temp_image, g_kernel);

    /* save_image(ref_image->pixel, width, height, file_save_blurred_ref); */
    /* save_image(obj_image->pixel, width, height, file_save_blurred_obj); */

    //Difference change obj
    difference(ref_image, obj_image);
    
    /* save_image(obj_image->pixel, width, height, diff_image); */

    struct MorphologicalKernel* k1 = circular_kernel(5);
    struct MorphologicalKernel* k2 = circular_kernel(11);
   
    // Perform closing
    perform_erosion(obj_image, temp_image, k1);
    perform_dilation(obj_image, temp_image, k1);
    
    /* save_image(obj_image->pixel, width, height, file_save_closing); */
   
    // Perform opening
    perform_dilation(obj_image, temp_image, k2);
    perform_erosion(obj_image, temp_image, k2);

    /* save_image(obj_image->pixel, width, height, file_save_opening); */

    int nb_components = compute_threshold(obj_image, temp_image);
    /* save_image(obj_image->pixel, width, height, file_save_threshold_base); */

    struct Bbox** bboxes = get_bbox(obj_image, nb_components);
    *nb_obj = nb_components;

    freeImageMat(ref_image);
    freeImageMat(temp_image);
    freeImageMat(obj_image);
    freeGaussianKernel(g_kernel);
    freeMorphologicalKernel(k1);
    freeMorphologicalKernel(k2);

    return bboxes;
}
