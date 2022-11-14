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

set(cudf_version 23.02.00)

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
option(USE_LIBARROW_FROM_PYARROW "Use the libarrow contained within pyarrow." OFF)
mark_as_advanced(USE_LIBARROW_FROM_PYARROW)

# If the user requested it we attempt to find CUDF.
if(FIND_CUDF_CPP)
  if(USE_LIBARROW_FROM_PYARROW)
    # We need to find arrow before libcudf since libcudf requires it but doesn't bundle it. TODO:
    # These options should probably all become optional since in practice they aren't meaningful
    # except in the case where we actually compile Arrow.
    set(CUDF_USE_ARROW_STATIC OFF)
    set(CUDF_ENABLE_ARROW_S3 OFF)
    set(CUDF_ENABLE_ARROW_ORC OFF)
    set(CUDF_ENABLE_ARROW_PYTHON OFF)
    set(CUDF_ENABLE_ARROW_PARQUET OFF)
    include(rapids-find)
    include(rapids-export)
    include(../../cpp/cmake/thirdparty/get_arrow.cmake)
  endif()

  find_package(cudf ${cudf_version} REQUIRED)
else()
  set(cudf_FOUND OFF)
endif()

include(rapids-cython)

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
  # library in the cudf directory as a single source of truth and modify the other rpaths
  # appropriately.
  set(cython_lib_dir cudf)
  install(TARGETS cudf DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(cudf/_lib)

include(cmake/Modules/ProtobufHelpers.cmake)
codegen_protoc(cudf/utils/metadata/orc_column_statistics.proto)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cudf PATHS "${cython_lib_dir}")
endif()
