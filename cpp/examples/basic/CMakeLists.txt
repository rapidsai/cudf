# Copyright (c) 2020-2023, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

project(
  basic_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(basic_example src/process_csv.cpp)
target_link_libraries(basic_example PRIVATE cudf::cudf)
target_compile_features(basic_example PRIVATE cxx_std_17)
