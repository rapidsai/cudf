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
include(rapids-cuda)
rapids_cuda_init_architectures(cudf-python)

project(
  cudf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(FIND_CUDF_CPP "Search for existing CUDF C++ installations before defaulting to local files"
       OFF
)

# Find Python early so that later commands can use it
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)

# If the user requested it we attempt to find CUDF.
if(FIND_CUDF_CPP)
  include(rapids-cpm)
  include(rapids-export)
  include(rapids-find)
  rapids_cpm_init()

  find_package(cudf "${RAPIDS_VERSION}" REQUIRED)

  # an installed version of libcudf doesn't provide the dlpack headers so we need to download dlpack
  # for the interop.pyx
  include(../../cpp/cmake/thirdparty/get_dlpack.cmake)
else()
  set(cudf_FOUND OFF)
endif()

include(rapids-cython-core)

if(NOT cudf_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
  set(CUDA_STATIC_RUNTIME ON)

  add_subdirectory(../../cpp cudf-cpp EXCLUDE_FROM_ALL)

  # libcudf targets are excluded by default above via EXCLUDE_FROM_ALL to remove extraneous
  # components like headers from libcudacxx, but we do need the libraries. However, we want to
  # control where they are installed to. Since there are multiple subpackages of cudf._lib that
  # require access to libcudf, we place the library and all its dependent artifacts in the cudf
  # directory as a single source of truth and modify the other rpaths appropriately.
  set(cython_lib_dir cudf)
  include(../pylibcudf/cmake/Modules/WheelHelpers.cmake)
  # TODO: This install is currently overzealous. We should only install the libraries that are
  # downloaded by CPM during the build, not libraries that were found on the system.  However, in
  # practice right this would only be a problem is if libcudf was not found but some of the
  # dependencies were, and we have no real use cases where that happens.
  install_aliased_imported_targets(
    TARGETS cudf arrow_shared nvcomp::nvcomp nvcomp::nvcomp_gdeflate nvcomp::nvcomp_bitcomp
    DESTINATION ${cython_lib_dir}
  )
endif()

rapids_cython_init()

add_subdirectory(cudf/_lib)
add_subdirectory(udf_cpp)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cudf PATHS "${cython_lib_dir}")
endif()
