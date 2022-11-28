#include "detect_obj_gpu.hpp"
#include <cassert>
#include <iostream>
#include "helpers_gpu.hpp"


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

    if (*b_init != current)
    {
        *has_change = true;
        *b_init = current;
    }
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

__global__ void apply_bin_threshold(unsigned int *buffer_bin, unsigned char *buffer_base, size_t rows, size_t cols,
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

bool get_has_change(bool *d_has_change) {
    bool h_has_change;
    cudaMemcpy(&h_has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);
    return h_has_change;
}

int connexe_components(unsigned char *buffer_base, size_t rows, size_t cols, size_t pitch, unsigned char threshold, int thx, int thy) {
    dim3 threads(thx, thy);
    dim3 blocks(std::ceil(float(cols) / float(threads.x)), std::ceil(float(rows) / float(threads.y)));
    
    size_t pitch_bin;
    unsigned int *buffer_bin = malloc2Dcuda<unsigned int>(rows, cols, &pitch_bin);

    apply_bin_threshold<<<blocks, threads>>>(buffer_bin, buffer_base, rows, cols, pitch, pitch_bin, threshold);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool *d_has_change = mallocCpy<bool>(false, sizeof(bool));
    bool h_has_change = true;

    while (h_has_change) {
	set_value<<<1, 1>>>(d_has_change, false);
        for (int i = 0; i < 5; i++) {
            propagate2<<<blocks, threads>>>(buffer_base, buffer_bin, rows, cols, pitch, pitch_bin, d_has_change);
        }
        cudaDeviceSynchronize();
        cudaCheckError();
	h_has_change = get_has_change(d_has_change);
    }

    int h_nb_compo = 1;
    int *d_nb_compo = mallocCpy<int>(1, sizeof(int));

    unsigned char *labels = malloc1Dcuda<unsigned char>(sizeof(unsigned char) * rows * cols);
    cudaMemset(labels, 0, rows * cols * sizeof(unsigned char));

    mask_label<<<blocks, threads>>>(buffer_bin, labels, rows, cols, pitch_bin);
    cudaDeviceSynchronize();
    cudaCheckError();
    
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
