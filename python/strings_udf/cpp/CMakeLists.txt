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

# depends on an installed cudf dev env
set(UDF_INCLUDES ${CONDA_PREFIX}/include)
list(APPEND UDF_INCLUDES ${CONDA_PREFIX}/include/rapids/libcudacxx)

set(UDF_LIBS ${CONDA_PREFIX}/lib/libcudf.so nvrtc)

set(UDF_CXX_FLAGS "")
set(UDF_CUDA_FLAGS "")
set(SHIM_CUDA_FLAGS "")

list(APPEND UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true -keep)

add_library(cudf_strings_udf SHARED src/strings/udf/udf_apis.cu)
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

set_target_properties(
  cudf_strings_udf
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             PUBLIC_HEADER "include/cudf/strings/udf/udf_apis.hpp"
)

target_compile_options(
  cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
)

target_compile_options(shim PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")

target_include_directories(
  cudf_strings_udf
  PRIVATE include
  PUBLIC ${UDF_INCLUDES}
)
target_link_libraries(cudf_strings_udf PUBLIC ${UDF_LIBS})

install(TARGETS cudf_strings_udf
        PUBLIC_HEADER DESTINATION ${CONDA_PREFIX}/include/cudf/strings/udf/
)
install(FILES ${CMAKE_SOURCE_DIR}/build/CMakeFiles/shim.dir/src/strings/udf/shim.ptx
        DESTINATION ${CONDA_PREFIX}/lib
)
