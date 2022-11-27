#include "utils_gpu.hpp"

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
