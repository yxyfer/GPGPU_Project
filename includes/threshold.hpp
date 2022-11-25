#pragma once

unsigned char otsu_threshold(struct ImageMat* in_image);

int connexe_components(struct ImageMat* img_1, struct ImageMat* temp);

