#pragma once

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

template <typename T>
T* malloc2Dcuda(size_t rows, size_t cols, size_t *pitch) {
    T *buffer_device;
    cudaMallocPitch(&buffer_device, pitch, sizeof(T) * cols, rows);
    cudaCheckError();
    return buffer_device;
}

template <typename T>
__device__ inline T* eltPtr(T *baseAddress, int col, int row, size_t pitch) {
    return (T*)((char*)baseAddress + row * pitch + col * sizeof(T));
}

template <class T>
T *mallocCpy(T value, size_t size) {
    T *device;
    cudaMalloc((void**) &device, size);
    cudaMemcpy(device, &value, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    return device;
}

template <class T>
T *malloc1Dcuda(size_t size) {
    T *device;
    cudaMalloc((void**) &device, size);
    cudaCheckError();

    return device;
}

