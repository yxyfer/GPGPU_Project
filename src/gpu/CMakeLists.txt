add_library(threshold_gpu)
target_compile_features(threshold_gpu PUBLIC cxx_std_14)
target_sources(threshold_gpu PRIVATE
threshold/threshold_gpu.cu
threshold/otsu_gpu.cu
threshold/connexe_components_gpu.cu
)

add_library(bbox_gpu)
target_compile_features(bbox_gpu PUBLIC cxx_std_14)
target_sources(bbox_gpu PRIVATE
    bbox_gpu.cu
)

add_library(gray_gpu)
target_compile_features(gray_gpu PUBLIC cxx_std_14)
target_sources(gray_gpu PRIVATE
    gray_scale_gpu.cu
)

add_library(blurr_gpu)
target_compile_features(blurr_gpu PUBLIC cxx_std_14)
target_sources(blurr_gpu PRIVATE
    blurr_gpu.cu
)

add_library(diff_gpu)
target_compile_features(diff_gpu PUBLIC cxx_std_14)
target_sources(diff_gpu PRIVATE
    difference_gpu.cu
)

add_library(helpers_gpu)
target_compile_features(helpers_gpu PUBLIC cxx_std_14)
target_sources(helpers_gpu PRIVATE
    helpers_gpu.cu
)
