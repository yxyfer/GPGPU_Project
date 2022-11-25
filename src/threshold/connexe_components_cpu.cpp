#include "../threshold.hpp"

#include "../detect_obj.hpp"
#include "../utils.hpp"

void components(struct ImageMat* img,
                struct ImageMat* temp,
                int x,
                int y,
                int val)
{
    if (x < 0 || x >= img->height || y < 0 || y >= img->width ||
        img->pixel[x][y] % 255 == 0)
        return;

    temp->pixel[x][y] = val;
    img->pixel[x][y] = 255;
    components(img, temp, x + 1, y, val);
    components(img, temp, x - 1, y, val);
    components(img, temp, x, y + 1, val);
    components(img, temp, x, y - 1, val);
}

int connexe_components(struct ImageMat* img_1, struct ImageMat* temp)
{
    int number = 1;
    for (int i = 0; i < img_1->height; i++) {
        for (int j = 0; j < img_1->width; j++) {
            if (temp->pixel[i][j] == 255) {
                img_1->pixel[i][j] = 1;
                components(img_1, temp, i, j, number);
                number++;
            }
        }
    }

    return number - 1;
}
