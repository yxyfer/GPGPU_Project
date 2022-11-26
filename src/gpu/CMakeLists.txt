add_library(threshold_gpu)
target_compile_features(threshold_gpu PUBLIC cxx_std_14)
target_sources(threshold_gpu PRIVATE
# threshold/gpu_functions.cu
threshold/threshold_gpu.cu
threshold/otsu_gpu.cu
threshold/connexe_components_gpu.cu
)
# set_target_properties(threshold_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
