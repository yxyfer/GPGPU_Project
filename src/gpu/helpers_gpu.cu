#include "helpers_gpu.hpp"


int getProperties(int device, cudaDeviceProp *deviceProp)
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
