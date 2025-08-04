# Copyright (c) 2022-2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(strings_examples)

project(
  strings_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(libcudf_apis libcudf_apis.cpp)
target_compile_features(libcudf_apis PRIVATE cxx_std_20)
target_link_libraries(libcudf_apis PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS libcudf_apis DESTINATION bin/examples/libcudf)

add_executable(custom_with_malloc custom_with_malloc.cu)
target_compile_features(custom_with_malloc PRIVATE cxx_std_20)
target_compile_options(custom_with_malloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(
  custom_with_malloc PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS custom_with_malloc DESTINATION bin/examples/libcudf)

add_executable(custom_prealloc custom_prealloc.cu)
target_compile_features(custom_prealloc PRIVATE cxx_std_20)
target_compile_options(custom_prealloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_prealloc PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS custom_prealloc DESTINATION bin/examples/libcudf)

add_executable(custom_optimized custom_optimized.cu)
target_compile_features(custom_optimized PRIVATE cxx_std_20)
target_compile_options(custom_optimized PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_optimized PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS custom_optimized DESTINATION bin/examples/libcudf)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/names.csv DESTINATION bin/examples/libcudf)
