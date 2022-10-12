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

add_library(detecter)
target_compile_features(detecter PUBLIC cxx_std_14)
target_link_libraries(detecter PRIVATE spdlog::spdlog)
target_sources(detecter PRIVATE src/detect_obj.hpp src/detect_obj_gpu.cu src/detect_obj_cpu.cpp)
set_target_properties(detecter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(blurring)
target_compile_features(blurring PUBLIC cxx_std_14)
target_link_libraries(blurring PRIVATE spdlog::spdlog)
target_sources(blurring PRIVATE src/detect_obj.hpp src/blur_cpu.cpp)

add_library(openclose)
target_compile_features(openclose PUBLIC cxx_std_14)
target_link_libraries(openclose PRIVATE spdlog::spdlog)
target_sources(openclose PRIVATE src/detect_obj.hpp src/opening.cpp src/opening_old.cpp)

add_library(threshold)
target_compile_features(threshold PUBLIC cxx_std_14)
target_link_libraries(threshold PRIVATE spdlog::spdlog)
target_sources(threshold PRIVATE src/detect_obj.hpp src/threshold_cpu.cpp)

# add_library(utils)
# target_compile_features(utils PUBLIC cxx_std_14)
# target_link_libraries(utils PRIVATE spdlog::spdlog)
# target_sources(utils PRIVATE src/utils.hpp src/utils.hxx)

add_executable(mandel src/mandel.cpp)
target_compile_features(mandel PUBLIC cxx_std_17)
target_link_libraries(mandel PRIVATE CLI11::CLI11 spdlog::spdlog detecter blurring openclose threshold) 
# target_link_libraries(mandel PRIVATE PNG::PNG CLI11::CLI11 spdlog::spdlog detecter)

add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE spdlog::spdlog benchmark::benchmark detecter)

