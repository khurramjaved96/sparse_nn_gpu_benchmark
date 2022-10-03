cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(CUDATest include/sequential_memory.cuh src/sequential_memory.cu sequential_memory_benchmark.cu)

add_executable(CUDATestRandom random_memory_access_benchmark.cu src/random_memory.cu include/random_memory.cuh )

target_link_libraries(CUDATest PUBLIC tbb)
target_link_libraries(CUDATestRandom PUBLIC tbb)

project(FlexibleNN LANGUAGES CXX)

