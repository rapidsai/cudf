# Copyright (c) 2024-2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(hybrid_scan)

project(
  hybrid_scan
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

add_library(hybrid_scan_utils OBJECT hybrid_scan_utils.cpp io_source.cpp)
target_compile_features(hybrid_scan_utils PRIVATE cxx_std_20)
target_link_libraries(hybrid_scan_utils PRIVATE cudf::cudf)

# Build and install hybrid_scan
add_executable(hybrid_scan hybrid_scan.cpp)
target_link_libraries(
  hybrid_scan PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
                      $<TARGET_OBJECTS:hybrid_scan_utils>
)
target_compile_features(hybrid_scan PRIVATE cxx_std_20)
install(TARGETS hybrid_scan DESTINATION bin/examples/libcudf)

# Install the example.parquet file
install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.parquet DESTINATION bin/examples/libcudf)
