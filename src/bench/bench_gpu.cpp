#include "helpers_images.hpp"
#include "detect_obj_gpu.hpp"

#include <benchmark/benchmark.h>
#include <iostream>

std::string ref_image_path = "../images/base.png";
std::string obj_image_path = "../images/obj.png";
int width, height, channels;

unsigned char* ref_image = load_image(const_cast<char*>(ref_image_path.c_str()),
                                      &width,
                                      &height,
                                      &channels);

unsigned char* obj_image = load_image(const_cast<char*>(obj_image_path.c_str()),
                                      &width,
                                      &height,
                                      &channels);

void BM_gray_scale_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    const size_t size_color = cols * rows * channels * sizeof(unsigned char);
    
    unsigned char *buffer_ref_cuda = cpyToCuda(ref_image, size_color);
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    
    for (auto _ : st) {
        gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, 32, 32);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_blurring_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    double *kernel_gpu = create_gaussian_kernel_gpu(5);

    for (auto _ : st)
        apply_blurr_gpu(gray_ref_cuda, rows, cols, 5, kernel_gpu, pitch, 32, 32);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_difference_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    unsigned char *gray_obj_cuda = initCuda(rows, cols, &pitch);
    
    for (auto _ : st)
        difference_gpu(gray_ref_cuda, gray_obj_cuda, rows, cols, pitch, 32, 32);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_closing_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    unsigned char *obj = initCuda(rows, cols, &pitch);
    
    size_t k1_size = 5;
    unsigned char *morpho_k1 = circular_kernel_gpu(k1_size);

    // Perform closing
    for (auto _ : st) {
        erosion_gpu(obj, rows, cols, k1_size, morpho_k1, pitch, 32, 32);
        dilation_gpu(obj, rows, cols, k1_size, morpho_k1, pitch, 32, 32);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_opening_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    unsigned char *obj = initCuda(rows, cols, &pitch);
    
    size_t k2_size = 11;
    unsigned char *morpho_k2 = circular_kernel_gpu(k2_size);

    // Perform closing
    for (auto _ : st) {
        dilation_gpu(obj, rows, cols, k2_size, morpho_k2, pitch, 32, 32);
        erosion_gpu(obj, rows, cols, k2_size, morpho_k2, pitch, 32, 32);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_first_threshold_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    const size_t size_color = cols * rows * channels * sizeof(unsigned char);
    
    unsigned char *buffer_ref_cuda = cpyToCuda(ref_image, size_color);
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    
    gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, 32, 32);

    for (auto _ : st) {
        threshold(gray_ref_cuda, rows, cols, pitch, 32, 32);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_connexe_components_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    const size_t size_color = cols * rows * channels * sizeof(unsigned char);
    
    unsigned char *buffer_ref_cuda = cpyToCuda(ref_image, size_color);
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    
    gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, 32, 32);

    unsigned char th = threshold(gray_ref_cuda, rows, cols, pitch, 32, 32);

    for (auto _ : st) {
        connexe_components(gray_ref_cuda, rows, cols, pitch, th, 32, 32);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_threshold_gpu(benchmark::State& st)
{
    size_t pitch;
    const int rows = height;
    const int cols = width;
    
    const size_t size_color = cols * rows * channels * sizeof(unsigned char);
    
    unsigned char *buffer_ref_cuda = cpyToCuda(ref_image, size_color);
    unsigned char *gray_ref_cuda = initCuda(rows, cols, &pitch);
    
    gray_scale_gpu(buffer_ref_cuda, gray_ref_cuda, rows, cols, pitch, channels, 32, 32);

    for (auto _ : st) {
        unsigned char th = threshold(gray_ref_cuda, rows, cols, pitch, 32, 32);
        connexe_components(gray_ref_cuda, rows, cols, pitch, th, 32, 32);
    }

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_main_gpu(benchmark::State& st)
{
    std::string ref_image_path = "../images/base.png";
    std::string obj_image_path = "../images/obj.png";

    int width, height, channels;
    unsigned char* ref_image = load_image(
        const_cast<char*>(ref_image_path.c_str()), &width, &height, &channels);
    unsigned char* obj_image = load_image(
        const_cast<char*>(obj_image_path.c_str()), &width, &height, &channels);

    unsigned char** images =
        (unsigned char**)std::malloc(sizeof(unsigned char*) * 2);
    images[0] = ref_image;
    images[1] = obj_image;

    int* nb_objs = (int*)std::malloc(1 * sizeof(int));

    for (auto _ : st)
        main_detection_gpu(images, 2, width, height, channels, nb_objs);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gray_scale_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_blurring_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_difference_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_closing_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_opening_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_first_threshold_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_connexe_components_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_threshold_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_main_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
