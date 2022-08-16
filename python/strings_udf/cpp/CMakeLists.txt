# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake
)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()
include(cmake/thirdparty/get_jitify.cmake)

# set the project name and version
project(
  cudf_strings_udf
  VERSION 0.1
  DESCRIPTION "cudf strings-udf library"
  LANGUAGES C CXX CUDA
)

if(NOT CONDA_PREFIX)
  message(FATAL_ERROR " CONDA_PREFIX must be set")
endif()

message(STATUS "CONDA_PREFIX=${CONDA_PREFIX}")

find_package(CUDAToolkit REQUIRED)

set(SHIM_CUDA_FLAGS "")

list(APPEND SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true -keep)

add_library(shim OBJECT src/strings/udf/shim.cu)

set_property(TARGET shim PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(
  shim
  PRIVATE include
  PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(
  shim
  PRIVATE include
  PUBLIC ${CONDA_PREFIX}/include
)

target_compile_options(shim PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")

install(FILES ${CMAKE_SOURCE_DIR}/build/CMakeFiles/shim.dir/src/strings/udf/shim.ptx
        DESTINATION ${CMAKE_SOURCE_DIR}/../strings_udf/shim/
)
