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

target_sources(detecter PRIVATE src/detect.hpp src/detect.cu)
set_target_properties(detecter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mandel src/mandel.cpp)
target_compile_features(mandel PUBLIC cxx_std_17)
target_link_libraries(mandel PRIVATE CLI11::CLI11 spdlog::spdlog detecter)
# target_link_libraries(mandel PRIVATE PNG::PNG CLI11::CLI11 spdlog::spdlog detecter)

add_executable(bench src/bench.cpp)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE spdlog::spdlog benchmark::benchmark detecter)

