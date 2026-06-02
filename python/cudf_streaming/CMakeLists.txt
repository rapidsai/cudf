# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(cudf-streaming-python)

project(
  cudf-streaming-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * find C++ libraries from RAPIDS wheels in site-packages -----------------------------------
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

find_package(cudf_streaming "${RAPIDS_VERSION}" REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cudf_streaming/integrations)
add_subdirectory(cudf_streaming/streaming)
