# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(hybrid_scan_io)

project(
  hybrid_scan_io
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

add_library(hybrid_scan_io_utils OBJECT common_utils.cpp io_source.cpp)
target_compile_features(hybrid_scan_io_utils PRIVATE cxx_std_20)
target_link_libraries(hybrid_scan_io_utils PRIVATE cudf::cudf)

# Build and install hybrid_scan
add_executable(hybrid_scan_io hybrid_scan_io.cpp)
target_link_libraries(
  hybrid_scan_io PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
                         $<TARGET_OBJECTS:hybrid_scan_io_utils>
)
target_compile_features(hybrid_scan_io PRIVATE cxx_std_20)
install(TARGETS hybrid_scan_io DESTINATION bin/examples/libcudf/hybrid_scan_io)

# Install the example.parquet file
install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.parquet
        DESTINATION bin/examples/libcudf/hybrid_scan_io
)
