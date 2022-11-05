#include "detect_obj.hpp"
#include "helpers_images.hpp"

#include <benchmark/benchmark.h>


void BM_gray_scale(benchmark::State& st) {
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    
    unsigned char *ref_image = load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    struct ImageMat *image = new_matrix(height, width);

    for (auto _ : st) {
        to_gray_scale(ref_image, image, width, height, 3);
    }
    
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_blurring(benchmark::State& st) {
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    
    load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    struct ImageMat *image = new_matrix(height, width);
    struct ImageMat *temp_image = new_matrix(height, width);
    
    struct GaussianKernel* g_kernel = create_gaussian_kernel(5);
    
    for (auto _ : st)
        apply_blurring(image, temp_image, g_kernel);
    
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_difference(benchmark::State& st) {
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    
    load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    struct ImageMat *image1 = new_matrix(height, width);
    struct ImageMat *image2 = new_matrix(height, width);
    
    for (auto _ : st)
        difference(image1, image2);
    
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_closing(benchmark::State& st) {
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    
    load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    struct MorphologicalKernel* k1 = circular_kernel(5);
    struct ImageMat *image1 = new_matrix(height, width);
    struct ImageMat *image2 = new_matrix(height, width);
    
    for (auto _ : st) {
        perform_erosion(image1, image2, k1);
        perform_dilation(image1, image2, k1);
    }
    
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_opening(benchmark::State& st) {
    int width, height, channels;
    std::string ref_image_path = "../images/blank.jpg";
    
    load_image(const_cast<char *>(ref_image_path.c_str()), &width, &height, &channels);
    struct MorphologicalKernel* k1 = circular_kernel(11);
    struct ImageMat *image1 = new_matrix(height, width);
    struct ImageMat *image2 = new_matrix(height, width);
    
    for (auto _ : st) {
        perform_dilation(image1, image2, k1);
        perform_erosion(image1, image2, k1);
    }
    
    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

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

BENCHMARK(BM_gray_scale)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_blurring)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_difference)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_closing)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_opening)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

/* BENCHMARK(BM_Rendering_gpu) */
/* ->Unit(benchmark::kMillisecond) */
/* ->UseRealTime(); */

BENCHMARK_MAIN();
