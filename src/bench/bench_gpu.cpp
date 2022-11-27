#include "detect_obj.hpp"
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

void BM_gray_scale(benchmark::State& st)
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


BENCHMARK(BM_gray_scale)->Unit(benchmark::kMillisecond)->UseRealTime();

/* BENCHMARK(BM_blurring)->Unit(benchmark::kMillisecond)->UseRealTime(); */

/* BENCHMARK(BM_difference)->Unit(benchmark::kMillisecond)->UseRealTime(); */

/* BENCHMARK(BM_closing)->Unit(benchmark::kMillisecond)->UseRealTime(); */

/* BENCHMARK(BM_opening)->Unit(benchmark::kMillisecond)->UseRealTime(); */

/* BENCHMARK(BM_threshold)->Unit(benchmark::kMillisecond)->UseRealTime(); */

/* BENCHMARK(BM_main_cpu)->Unit(benchmark::kMillisecond)->UseRealTime(); */
/* BENCHMARK(BM_Rendering_gpu) */
/* ->Unit(benchmark::kMillisecond) */
/* ->UseRealTime(); */

BENCHMARK_MAIN();
