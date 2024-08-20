# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
  libcudf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Check if cudf is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cudf package.
find_package(cudf "${RAPIDS_VERSION}")

if(cudf_FOUND)
  return()
endif()

unset(cudf_FOUND)

# For wheels, this should always be true
set(USE_LIBARROW_FROM_PYARROW ON)

# Find Python early so that later commands can use it
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
set(CUDA_STATIC_RUNTIME ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

include(../pylibcudf/cmake/Modules/LinkPyarrowHeaders.cmake)

add_subdirectory(../../cpp cudf-cpp)

# Ensure other libraries needed by libcudf.so get installed alongside it.
#
# TODO: This install is currently overzealous. We should only install the libraries that are
# downloaded by CPM during the build, not libraries that were found on the system.
include(cmake/Modules/WheelHelpers.cmake)
install_aliased_imported_targets(
  TARGETS cudf arrow_shared nvcomp::nvcomp nvcomp::nvcomp_gdeflate nvcomp::nvcomp_bitcomp
  DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)
