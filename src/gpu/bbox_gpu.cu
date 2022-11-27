#include "detect_obj_gpu.hpp"

__global__ void bbox(unsigned char *buffer, int *maxh, int *minh, int *maxw, int *minw, size_t rows, size_t cols, size_t pitch){

   int col = blockDim.x * blockIdx.x + threadIdx.x;
   int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows) {
        return;
    }

    unsigned char myval = buffer[col + row * pitch];
  
    if (myval > 0) {
      atomicMax(maxw + myval - 1, col);
      atomicMin(minw + myval - 1, col);
      atomicMax(maxh + myval - 1, row);
      atomicMin(minh + myval - 1, row);
    }
}

void get_bbox(unsigned char *buffer, size_t rows, size_t cols, size_t pitch, int nb_components) {
    int max = nb_components;
    int *maxw, *maxh, *minw, *minh, *d_maxw, *d_maxh, *d_minw, *d_minh;
   
    maxw = new int[max];
    maxh = new int[max];
    minw = new int[max];
    minh = new int[max];
  
    cudaMalloc(&d_maxw, max * sizeof(int));
    cudaMalloc(&d_maxh, max * sizeof(int));
    cudaMalloc(&d_minw, max * sizeof(int));
    cudaMalloc(&d_minh, max * sizeof(int));
  
    for (int i = 0; i < max; i++) {
        maxw[i] = 0;
        maxh[i] = 0;
        minw[i] = cols;
        minh[i] = rows;
    }

    cudaMemcpy(d_maxw, maxw, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxh, maxh, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minw, minw, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minh, minh, max * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    bbox<<<blocks, threads>>>(buffer, d_maxh, d_minh, d_maxw, d_minw, rows, cols, pitch);
  
    cudaMemcpy(maxw, d_maxw, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxh, d_maxh, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minw, d_minw, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minh, d_minh, max * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < max; i++)
        printf("label %d, maxh: %d, minh: %d, maxw: %d, minw: %d\n", i + 1, maxh[i], minh[i], maxw[i], minw[i]);
}
