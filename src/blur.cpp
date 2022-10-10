#include "blur.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

unsigned int convolution(int x, int y, unsigned char **image, unsigned char** kernel, unsigned char kernel_size)
{
    unsigned int conv = 0;
    // get kernel val
    int margin = (int) kernel_size / 2;
    for(int i = -margin ; i <= margin; i++)
        for(int j = -margin; j <= margin; j++)
            conv += image[x + i][y + j] * kernel[i + margin][j + margin];
    return conv;
}

unsigned int divisor(unsigned char** kernel, unsigned char kernel_size) {
    unsigned int div = 0;
    // Calculate weights
    for(int i = 0; i < kernel_size; i++)
        for(int j = 0; j < kernel_size; j++)
            div += kernel[i][j];

    return div;
}


// Apply gaussian blur to the image
unsigned char **blurring(unsigned char** image, unsigned char** kernel, int width, int height, unsigned char kernel_size) {
    // Allocating for blurred image
    unsigned char** blurred_image = (unsigned char**) malloc(height * sizeof(unsigned char*));
    for(int i = 0; i < height; i++)
        blurred_image[i] = (unsigned char*) malloc(width * sizeof(unsigned char));


    int margin = (int) kernel_size / 2;
    unsigned int div = divisor(kernel, kernel_size);
    // TODO: MANAGE LES EDGES !!!!! 
    // TODO: CREATE A KERNEL GENERATOR
    for (int x = margin; x < height - margin; ++x)
        for (int y = margin; y < width - margin; ++y) {
            blurred_image[x][y] = convolution(x, y, image, kernel, kernel_size) / div;
        }
    

   return blurred_image; 
}

