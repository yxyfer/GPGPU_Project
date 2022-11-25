#pragma once
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

template <typename T>
T *malloc_cuda(size_t size)
{
    T *buffer;
    cudaMalloc((void **)&buffer, size);
    cudaCheckError();

    return buffer;
}

template <typename T>
T *cpy_host_to_device(T *buffer, size_t size)
{
    T *device_buffer = malloc_cuda<T>(size);

    cudaMemcpy(device_buffer, buffer, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    return device_buffer;
}

unsigned char *malloc_cuda2D(size_t cols, size_t rows, size_t *pitch)
{
    unsigned char *buffer;
    // Allocate an 2D buffer with padding
    //@@ use cudaMallocPitch to allocate this buffer
    cudaMallocPitch(&buffer, pitch, cols * sizeof(unsigned char), rows);
    cudaCheckError();

    return buffer;
}

int get_properties(int device, cudaDeviceProp *deviceProp)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (device >= deviceCount)
        return -1;

    cudaGetDeviceProperties(deviceProp, device);
    if (deviceProp->major == 9999 && deviceProp->minor == 9999)
    {
        std::cerr << "No CUDA GPU has been detected" << std::endl;
        return -1;
    }
    return 0;
}
