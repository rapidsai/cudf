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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(cudf_version 22.08.00)

include(../../fetch_rapids.cmake)

project(
  cudf-python
  VERSION ${cudf_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
)

option(FIND_CUDF_CPP "Search for existing CUDF C++ installations before defaulting to local files"
       OFF
)

# If the user requested it we attempt to find CUDF.
if(FIND_CUDF_CPP)
  find_package(cudf ${cudf_version} REQUIRED)
else()
  set(cudf_FOUND OFF)
endif()

if(NOT cudf_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
  # languages for the C++ project even if this project does not require those languges.
  include(rapids-cuda)
  rapids_cuda_init_architectures(cudf-python)
  enable_language(CUDA)
  # Since cudf only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  include("${CMAKE_PROJECT_cudf-python_INCLUDE}")

  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  add_subdirectory(../../cpp cudf-cpp)

  # Since there are multiple subpackages of cudf._lib that require access to libcudf, we place the
  # library in the _lib/cpp directory as a single source of truth and modify the other rpaths
  # appropriately.
  install(TARGETS cudf DESTINATION cudf/_lib/cpp)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(cudf/_lib)
