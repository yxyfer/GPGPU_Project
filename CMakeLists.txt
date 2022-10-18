cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)


FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )


FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
# find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_library(utilsimage)
target_compile_features(utilsimage PUBLIC cxx_std_14)
target_sources(utilsimage PRIVATE src/stb_image.h src/stb_image_write.h src/helpers_images.hpp src/helpers_images.cpp)


##### CPU
add_library(blurring_cpu)
target_compile_features(blurring_cpu PUBLIC cxx_std_14)
target_sources(blurring_cpu PRIVATE src/detect_obj.hpp src/blur_cpu.cpp)

add_library(openclose_cpu)
target_compile_features(openclose_cpu PUBLIC cxx_std_14)
target_sources(openclose_cpu PRIVATE src/detect_obj.hpp src/opening.cpp)

add_library(threshold_cpu)
target_compile_features(threshold_cpu PUBLIC cxx_std_14)
target_sources(threshold_cpu PRIVATE src/detect_obj.hpp src/threshold_cpu.cpp)

add_library(detector_cpu)
target_compile_features(detector_cpu PUBLIC cxx_std_14)
target_link_libraries(detector_cpu PRIVATE spdlog::spdlog utilsimage blurring_cpu openclose_cpu threshold_cpu)
target_sources(detector_cpu PRIVATE src/detect_obj.hpp src/detect_obj_cpu.cpp)


##### GPU
add_library(detector_gpu)
target_compile_features(detector_gpu PUBLIC cxx_std_14)
target_link_libraries(detector_gpu PRIVATE spdlog::spdlog utilsimage)
target_sources(detector_gpu PRIVATE src/detect_obj.hpp src/detect_obj_gpu.cu)
set_target_properties(detector_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


##### MAIN
# add_executable(main src/mandel.cpp)
# target_compile_features(main PUBLIC cxx_std_17)
# target_link_libraries(main PRIVATE CLI11::CLI11 spdlog::spdlog detector_cpu detector_gpu utilsimage)

add_executable(main_cpu src/main_cpu.cpp)
target_compile_features(main_cpu PUBLIC cxx_std_17)
target_link_libraries(main_cpu PRIVATE CLI11::CLI11 spdlog::spdlog detector_cpu utilsimage)

add_executable(main_gpu src/main_gpu.cpp)
target_compile_features(main_gpu PUBLIC cxx_std_17)
target_link_libraries(main_gpu PRIVATE CLI11::CLI11 spdlog::spdlog detector_gpu utilsimage)

##### BENCH
add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE benchmark::benchmark detector_cpu utilsimage)
