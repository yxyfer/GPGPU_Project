#include "threshold_gpu.hpp"
#include <cassert>
#include <iostream>

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

__global__ void propagate(unsigned char *buffer_base, unsigned char *buffer_bin,
                          size_t rows, size_t cols, size_t pitch, bool *has_change) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (col >= cols || row >= rows || buffer_bin[col + row * pitch] != 1)
         return;

    bool change = false;

    if (col + 1 < cols && buffer_base[(col + 1) + row * pitch] != 0 && buffer_bin[(col + 1) + row * pitch] != 1) {
	    buffer_bin[(col + 1) + row * pitch] = 1;
	    change = true;
    }
    if (col - 1 >= 0 && buffer_base[(col - 1) + row * pitch] != 0 && buffer_bin[(col - 1) + row * pitch] != 1) {
	    buffer_bin[(col - 1) + row * pitch] = 1;
	    change = true;
    }
    if (row + 1 < rows && buffer_base[col + (row + 1) * pitch] != 0 && buffer_bin[col + (row + 1) * pitch] != 1) {
	    buffer_bin[col + (row + 1) * pitch] = 1;
	    change = true;
    }
    if (row - 1 >= 0 && buffer_base[col + (row - 1) * pitch] != 0 && buffer_bin[col + (row - 1) * pitch] != 1) {
	    buffer_bin[col + (row - 1) * pitch] = 1;
	    change = true;
    }
    
    if (change)
        *has_change = true;
}

__global__ void set_value(bool *has_change, bool val) {
	*has_change = val;
}

bool get_has_change(bool *d_has_change) {
	bool h_has_change;
    	cudaMemcpy(&h_has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);
	return h_has_change;
}

template <class T>
T *mallocCpy(T value) {
    T *device;
    cudaMalloc((void**) &device, sizeof(T));
    cudaMemcpy(device, &value, sizeof(T), cudaMemcpyHostToDevice);

    return device;
}

void connexe_components(unsigned char *buffer_base, unsigned char *buffer_bin,
                        size_t rows, size_t cols, size_t pitch, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));

    bool *d_has_change = mallocCpy<bool>(false);
    bool h_has_change = true;

    while (h_has_change) {
	set_value<<<1, 1>>>(d_has_change, false);
        for (int i = 0; i < 3; i++)
            propagate<<<blocks, threads>>>(buffer_base, buffer_bin, rows, cols, pitch, d_has_change);
	cudaCheckError();
  	cudaDeviceSynchronize();
	h_has_change = get_has_change(d_has_change);
    }
}
