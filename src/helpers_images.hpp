#pragma once
#include <string>

unsigned char *load_image(char *path, int *width, int *height, int *channels);

unsigned char **get_images(int argc, char **argv, int *width, int *height, int *channels);

void save(unsigned char *image, int width, int height, std::string filename);

void save_image(unsigned char **image, int width, int height, std::string filename);

void free_image(unsigned char* image); 

void free_images(unsigned char **images, int length);
