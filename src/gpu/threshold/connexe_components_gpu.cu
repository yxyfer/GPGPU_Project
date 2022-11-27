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

template <typename T>
__device__ inline T* eltPtr(T *baseAddress, int col, int row, size_t pitch) {
    return (T*)((char*)baseAddress + row * pitch + col * sizeof(T));
}

__global__ void propagate2(unsigned char *buffer_base, unsigned int *buffer_bin,
                          size_t rows, size_t cols, size_t pitch, size_t pitch_bin, bool *has_change) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows || *eltPtr<unsigned char>(buffer_base, col, row, pitch) == 0)
        return;

    unsigned int* b_init = eltPtr<unsigned int>(buffer_bin, col, row, pitch_bin);
    unsigned int current = *b_init;


    if (col + 1 < cols && *eltPtr<unsigned int>(buffer_bin, col + 1, row, pitch_bin) != 0)
    {
        unsigned int val = *eltPtr<unsigned int>(buffer_bin, col + 1, row, pitch_bin);
        current = current == 0 ? val : min(current, val);
    }
    if (row + 1 < rows && *eltPtr<unsigned int>(buffer_bin, col, row + 1, pitch_bin) != 0)
    {
        unsigned int val = *eltPtr<unsigned int>(buffer_bin, col, row + 1, pitch_bin);
        current = current == 0 ? val : min(current, val);
    }
    if (col - 1 < cols && *eltPtr<unsigned int>(buffer_bin, col - 1, row, pitch_bin) != 0)
    {
        unsigned int val = *eltPtr<unsigned int>(buffer_bin, col - 1, row, pitch_bin);
        current = current == 0 ? val : min(current, val);
    }
    if (row - 1 < rows && *eltPtr<unsigned int>(buffer_bin, col, row - 1, pitch_bin) != 0)
    {
        unsigned int val = *eltPtr<unsigned int>(buffer_bin, col, row - 1, pitch_bin);
        current = current == 0 ? val : min(current, val);
    }

    __syncthreads();
    
    if (*b_init != current)
    {
        *has_change = true;
        *b_init = current;
    }

    /* printf("%d ", *b_init); */
}

__global__ void mask_label(unsigned int *buffer_bin, unsigned char *labelled, size_t rows, size_t cols, size_t pitch_bin) {

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    unsigned int* bin = eltPtr<unsigned int>(buffer_bin, col, row, pitch_bin);
    if (*bin == 0)
        return;

    unsigned int v = *bin;
    if (labelled[v] == (unsigned char) 0) {
        labelled[v] = (unsigned char) 1;
    }
}

__global__ void continous_labels(unsigned char *labels, size_t rows, size_t cols, int *val) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows || labels[col + row * cols] == 0)
        return;
    
    int old = atomicAdd(val, 1);
    labels[col + row * cols] = old;
}

__global__ void relabelled(unsigned char *buffer, unsigned int *buffer_bin, unsigned char *labelled,
                           size_t rows, size_t cols, size_t pitch, size_t pitch_bin) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    unsigned int* bin = eltPtr<unsigned int>(buffer_bin, col, row, pitch_bin);
    unsigned char* buf = eltPtr<unsigned char>(buffer, col, row, pitch);

    if (*bin == 0)
        *buf = 0;
    else {
        *buf = labelled[*bin];
    }
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

template <class T>
T *mallocSimple(size_t size) {
    T *device;
    cudaMalloc((void**) &device, size);

    return device;
}

template <class T>
T *mallocCpy(T value, size_t size) {
    T *device;
    cudaMalloc((void**) &device, size);
    cudaMemcpy(device, &value, size, cudaMemcpyHostToDevice);

    return device;
}

__global__ void set_val(unsigned char *buffer, size_t rows, size_t cols, unsigned char val) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    buffer[col + row * cols] = val;
}

__global__ void print_labels(unsigned char *buffer, size_t rows, size_t cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    printf("%d ", buffer[col + row * cols]);
}

__global__ void apply_bin_threshold2(unsigned int *buffer_bin, unsigned char *buffer_base, size_t rows, size_t cols,
                                     size_t pitch, size_t pitch_bin, int threshold) {
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= cols || row >= rows)
        return;

    unsigned int val = col + row * cols + 1;
    unsigned int *b_bin = (unsigned int *)((char*)buffer_bin + row * pitch_bin + col * sizeof(unsigned int));

    if (buffer_base[col + row * pitch] >= threshold)
        *b_bin = val;
    else
        *b_bin = 0;
}

template <typename T>
T* malloc2Dcuda(size_t rows, size_t cols, size_t *pitch) {
    T *buffer_device;
    cudaMallocPitch(&buffer_device, pitch, sizeof(T) * cols, rows);
    return buffer_device;
}

int connexe_components(unsigned char *buffer_base, size_t rows, size_t cols, size_t pitch, unsigned char threshold, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)),
                std::ceil(float(rows) / float(threads.y)));
    
    size_t pitch_bin;
    unsigned int *buffer_bin = malloc2Dcuda<unsigned int>(rows, cols, &pitch_bin);

    apply_bin_threshold2<<<blocks, threads>>>(buffer_bin, buffer_base, rows, cols, pitch, pitch_bin, threshold);
    cudaDeviceSynchronize();
    cudaCheckError();

    /* connexe_components(buffer, buffer_bin, rows, cols, pitch, pitch_bin, thx, thy); */

    bool *d_has_change = mallocCpy<bool>(false);
    bool h_has_change = true;

    while (h_has_change) {
	set_value<<<1, 1>>>(d_has_change, false);
        for (int i = 0; i < 3; i++) {
            propagate2<<<blocks, threads>>>(buffer_base, buffer_bin, rows, cols, pitch, pitch_bin, d_has_change);
            cudaDeviceSynchronize();
            cudaCheckError();
        }
	h_has_change = get_has_change(d_has_change);
    }

    int h_nb_compo = 1;
    int *d_nb_compo = mallocCpy<int>(1, sizeof(int));
    unsigned char *labels = mallocSimple<unsigned char>(sizeof(unsigned char) * rows * cols);
    set_val<<<blocks, threads>>>(labels, rows, cols, 0);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    mask_label<<<blocks, threads>>>(buffer_bin, labels, rows, cols, pitch_bin);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    /* print_labels<<<blocks, threads>>>(labels, rows, cols); */
    /* cudaDeviceSynchronize(); */
    /* cudaCheckError(); */
    

    continous_labels<<<blocks, threads>>>(labels, rows, cols, d_nb_compo);
    cudaDeviceSynchronize();
    cudaCheckError();

    relabelled<<<blocks, threads>>>(buffer_base, buffer_bin, labels, rows, cols, pitch, pitch_bin);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    cudaMemcpy(&h_nb_compo, d_nb_compo, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nb_compo);
    cudaFree(labels);
    cudaFree(buffer_bin);

    return h_nb_compo - 1;
}
