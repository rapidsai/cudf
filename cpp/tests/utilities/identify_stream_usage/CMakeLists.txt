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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.12/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)

project(
  IDENTIFY_STREAM_USAGE
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(rapids-cpm)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_init()
rapids_cpm_rmm()

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
add_library(identify_stream_usage SHARED identify_stream_usage.cpp)

find_package(CUDAToolkit REQUIRED)

set_target_properties(identify_stream_usage PROPERTIES CUDA_RUNTIME_LIBRARY SHARED)
target_link_libraries(identify_stream_usage PUBLIC CUDA::cudart rmm::rmm)

set_target_properties(
  identify_stream_usage
  PROPERTIES # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)

# Add the test file.
include(CTest)

add_executable(Tests test_default_stream_identification.cu)
add_test(NAME default_stream_identification COMMAND Tests)

set_tests_properties(
  default_stream_identification PROPERTIES ENVIRONMENT
                                           LD_PRELOAD=$<TARGET_FILE:identify_stream_usage>
)
