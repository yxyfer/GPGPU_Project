#include "detect_obj.hpp"
#include "helpers_images.hpp"

#include <benchmark/benchmark.h>

void BM_Rendering_cpu(benchmark::State& st)
{
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    std::string obj_image_path = "../images/object.jpg";

    unsigned char *ref_image = load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    unsigned char *obj_image = load_image(const_cast<char *>(obj_image_path.c_str()), &width, &height, &channels);

    for (auto _ : st)
        detect_cpu(ref_image, obj_image, width, height, channels);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

/* void BM_Rendering_gpu(benchmark::State& st) */
/* { */
/*   int stride = width * kRGBASize; */
/*   std::vector<char> data(height * stride); */

/*   /1* for (auto _ : st) *1/ */
/*   /1*   detect_gpu(data.data(), data.data(), width, height, stride); *1/ */

/*   st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate); */
/* } */

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

/* BENCHMARK(BM_Rendering_gpu) */
/* ->Unit(benchmark::kMillisecond) */
/* ->UseRealTime(); */

BENCHMARK_MAIN();
