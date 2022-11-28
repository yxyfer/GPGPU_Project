#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"

double *create_gaussian_kernel_gpu(unsigned char size)
{
    double *kernel = (double *)malloc(size * size * sizeof(double));
    unsigned char margin = size / 2;
    double sigma = 1.0;

    double s = 2.0 * sigma * sigma;
    // sum is for normalization
    double sum = 0.0;

    for (int row = -margin; row <= margin; row++)
    {
        for (int col = -margin; col <= margin; col++)
        {
            const double radius = col * col + row * row;
            kernel[(row + margin) * size + (col + margin)] =
                (exp(-radius / s)) / (M_PI * s);
            sum += kernel[(row + margin) * size + (col + margin)];
        }
    }

    // normalising the Kernel
    for (unsigned char i = 0; i < size; ++i)
        for (unsigned char j = 0; j < size; ++j)
            kernel[i * size + j] /= sum;

    size_t size_kernel_gpu = size * size * sizeof(double);
    return cpyHostToDevice<double>(kernel, size_kernel_gpu);
}

__global__ void gaussian_blur_gpu(unsigned char *image, int rows, int cols,
                                  int kernel_size, double *kernel, int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int start_k = kernel_size / 2;

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                res += (image[(y + j) * pitch + (x + i)]
                        * kernel[(j + start_k) * kernel_size + (i + start_k)]);
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

void apply_blurr_gpu(unsigned char *buffer, size_t rows, size_t cols, unsigned int kernel_size,
        double *kernel_gpu, size_t pitch, int thx, int thy) {
    const dim3 threads(thx, thy);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)), std::ceil(float(rows) / float(threads.y)));

    gaussian_blur_gpu<<<blocks, threads>>>(buffer, rows, cols, kernel_size, kernel_gpu, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();
}
