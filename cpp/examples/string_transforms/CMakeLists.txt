# Copyright (c) 2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(string_transforms_examples)

project(
  string_transforms_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an
# sccache bug in the version RAPIDS uses in CI that causes it to handle the
# resulting -M* flags incorrectly with gcc>=14. We can remove this once we
# upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(int_output int_output.cpp)
target_compile_features(int_output PRIVATE cxx_std_20)
target_compile_options(int_output PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(int_output PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS int_output DESTINATION bin/examples/libcudf)

add_executable(output output.cpp)
target_compile_features(output PRIVATE cxx_std_20)
target_compile_options(output PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(output PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS output DESTINATION bin/examples/libcudf)

add_executable(preallocated preallocated.cpp)
target_compile_features(preallocated PRIVATE cxx_std_20)
target_compile_options(preallocated PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(preallocated PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS preallocated DESTINATION bin/examples/libcudf)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/info.csv DESTINATION bin/examples/libcudf)
