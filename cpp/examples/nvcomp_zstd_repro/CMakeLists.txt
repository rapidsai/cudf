# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(nvcomp_zstd_repro)

project(
  nvcomp_zstd_repro
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(nvcomp_zstd_repro reproducer.cpp)
target_link_libraries(nvcomp_zstd_repro PRIVATE cudf::cudf)
target_compile_features(nvcomp_zstd_repro PRIVATE cxx_std_20)

install(TARGETS nvcomp_zstd_repro DESTINATION bin/examples/libcudf/nvcomp_zstd_repro)
