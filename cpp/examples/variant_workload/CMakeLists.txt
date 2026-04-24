# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(variant_workload_example)

project(
  variant_workload_example
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

# Configure your project here.  The workload-fixture header lives next to variant_workload.cpp.  The
# blob-builder utility header (cpp/benchmarks/io/variant_blob_builders.hpp) is shared with
# VARIANT_EXTRACT_NVBENCH and is included via relative path.
add_executable(variant_workload_example variant_workload.cpp)
target_link_libraries(variant_workload_example PRIVATE cudf::cudf)
target_compile_features(variant_workload_example PRIVATE cxx_std_20)

install(TARGETS variant_workload_example DESTINATION bin/examples/libcudf/variant_workload)
