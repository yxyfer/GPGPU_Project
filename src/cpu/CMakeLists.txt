add_library(blurring_cpu)
target_compile_features(blurring_cpu PUBLIC cxx_std_14)
target_sources(blurring_cpu PRIVATE
blur_cpu.cpp
)


add_library(openclose_cpu)
target_compile_features(openclose_cpu PUBLIC cxx_std_14)
target_sources(openclose_cpu PRIVATE
opening.cpp
)


add_library(threshold_cpu)
target_compile_features(threshold_cpu PUBLIC cxx_std_14)
target_sources(threshold_cpu PRIVATE
threshold/threshold_cpu.cpp
threshold/connexe_components_cpu.cpp
threshold/otsu_cpu.cpp
)

add_library(bbox_cpu)
target_compile_features(bbox_cpu PUBLIC cxx_std_14)
target_sources(bbox_cpu PRIVATE 
bbox.cpp
)
