# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

project(
  cudf-kafka-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

find_package(cudf_kafka "${RAPIDS_VERSION}" REQUIRED)

if(NOT cudf_kafka_FOUND)
  message(
    FATAL_ERROR
      "cudf_kafka package not found. cudf_kafka C++ is required to build this Python package."
  )
endif()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cudf_kafka/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cudf_kafka PATHS "${cython_lib_dir}")
endif()
