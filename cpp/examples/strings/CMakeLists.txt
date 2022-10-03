# Copyright (c) 2022, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.18)

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

set(CUDF_TAG branch-22.12)
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

add_executable(custom_gather custom_gather.cu)
target_compile_features(custom_gather PRIVATE cxx_std_17)
target_compile_options(custom_gather PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_gather PRIVATE cudf::cudf nvToolsExt)

add_executable(custom_fast custom_fast.cu)
target_compile_features(custom_fast PRIVATE cxx_std_17)
target_compile_options(custom_fast PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_fast PRIVATE cudf::cudf nvToolsExt)
