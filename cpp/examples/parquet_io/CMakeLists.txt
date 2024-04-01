# Copyright (c) 2020-2023, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

project(
  parquet_io
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(parquet_io parquet_io.cpp)
target_link_libraries(parquet_io PRIVATE cudf::cudf)
target_compile_features(parquet_io PRIVATE cxx_std_17)
