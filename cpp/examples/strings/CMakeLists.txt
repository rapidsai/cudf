# Copyright (c) 2022, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.23.1)

project(
  strings_examples
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

set(CUDF_TAG branch-23.04)
CPMFindPackage(
  NAME cudf GIT_REPOSITORY https://github.com/rapidsai/cudf
  GIT_TAG ${CUDF_TAG}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

#
add_executable(libcudf_apis libcudf_apis.cpp)
target_compile_features(libcudf_apis PRIVATE cxx_std_17)
target_link_libraries(libcudf_apis PRIVATE cudf::cudf nvToolsExt)

add_executable(custom_with_malloc custom_with_malloc.cu)
target_compile_features(custom_with_malloc PRIVATE cxx_std_17)
target_compile_options(custom_with_malloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_with_malloc PRIVATE cudf::cudf nvToolsExt)

add_executable(custom_prealloc custom_prealloc.cu)
target_compile_features(custom_prealloc PRIVATE cxx_std_17)
target_compile_options(custom_prealloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_prealloc PRIVATE cudf::cudf nvToolsExt)

add_executable(custom_optimized custom_optimized.cu)
target_compile_features(custom_optimized PRIVATE cxx_std_17)
target_compile_options(custom_optimized PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_optimized PRIVATE cudf::cudf nvToolsExt)
