cmake_minimum_required(VERSION 3.18)
project(continuous_batching_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 / 4070 SUPER = Ada Lovelace, sm_89

find_package(CUDAToolkit REQUIRED)

add_library(kernels STATIC kernels/linear.cu kernels/elementwise.cu kernels/attention.cu)
target_include_directories(kernels PUBLIC include)
target_link_libraries(kernels PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(core STATIC src/paged_kv_cache.cpp src/model_weights.cpp src/transformer.cpp)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC kernels)

add_executable(correctness_check src/correctness_check.cpp)
target_link_libraries(correctness_check PRIVATE core)

add_executable(bench_batching src/bench_batching.cpp)
target_link_libraries(bench_batching PRIVATE core)

find_package(Threads REQUIRED)
target_link_libraries(bench_batching PRIVATE Threads::Threads)

add_executable(correctness_test tests/correctness_test.cpp)
target_link_libraries(correctness_test PRIVATE core)

add_executable(server_demo src/server_demo.cpp)
target_link_libraries(server_demo PRIVATE core Threads::Threads)
