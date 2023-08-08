# Copyright (c) 2023, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

project(
  nested_types
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

set(CPM_DOWNLOAD_VERSION v0.35.3)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

set(CUDF_TAG branch-23.10)
CPMFindPackage(
  NAME cudf GIT_REPOSITORY https://github.com/rapidsai/cudf
  GIT_TAG ${CUDF_TAG}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)

# Configure your project here
add_executable(deduplication deduplication.cpp)
target_link_libraries(deduplication PRIVATE cudf::cudf)
target_compile_features(deduplication PRIVATE cxx_std_17)
