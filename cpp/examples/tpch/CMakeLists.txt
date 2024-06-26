# Copyright (c) 2020-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(tpch_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  tpch_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(tpch_q1 q1.cpp)
target_link_libraries(tpch_q1 PRIVATE cudf::cudf)
target_compile_features(tpch_q1 PRIVATE cxx_std_17)

add_executable(tpch_q6 q6.cpp)
target_link_libraries(tpch_q6 PRIVATE cudf::cudf)
target_compile_features(tpch_q6 PRIVATE cxx_std_17)