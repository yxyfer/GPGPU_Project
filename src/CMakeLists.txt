add_subdirectory(cpu)

# TO add when gpu implemented
# add_subdirectory(gpu)


add_library(bbox)
target_compile_features(bbox PUBLIC cxx_std_14)
target_sources(bbox PRIVATE 
bbox.cpp
)
