# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

set(cudf_version 23.08.00)

include(../../fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cudf-python)

project(
  cudf-python
  VERSION ${cudf_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

option(FIND_CUDF_CPP "Search for existing CUDF C++ installations before defaulting to local files"
       OFF
)
option(CUDF_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)
option(USE_LIBARROW_FROM_PYARROW "Use the libarrow contained within pyarrow." OFF)
mark_as_advanced(USE_LIBARROW_FROM_PYARROW)

# Always build wheels against the pyarrow libarrow.
if(CUDF_BUILD_WHEELS)
  set(USE_LIBARROW_FROM_PYARROW ON)
endif()

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

  # an installed version of libcudf doesn't provide the dlpack headers so we need to download dlpack
  # for the interop.pyx
  include(rapids-cpm)
  rapids_cpm_init()
  include(../../cpp/cmake/thirdparty/get_dlpack.cmake)
else()
  set(cudf_FOUND OFF)
endif()

include(rapids-cython)

if(NOT cudf_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  set(_exclude_from_all "")
  if(CUDF_BUILD_WHEELS)
    # We don't build C++ tests when building wheels, so we can also omit the test util and shrink
    # the wheel by avoiding embedding GTest.
    set(CUDF_BUILD_TESTUTIL OFF)
    set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)

    # Statically link cudart if building wheels
    set(CUDA_STATIC_RUNTIME ON)

    # Need to set this so all the nvcomp targets are global, not only nvcomp::nvcomp
    # https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_PACKAGE_TARGETS_GLOBAL.html#variable:CMAKE_FIND_PACKAGE_TARGETS_GLOBAL
    set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

    # Don't install the cuDF C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp cudf-cpp ${_exclude_from_all})

  if(CUDF_BUILD_WHEELS)
    include(cmake/Modules/WheelHelpers.cmake)
    get_target_property(_nvcomp_link_libs nvcomp::nvcomp INTERFACE_LINK_LIBRARIES)
    # Ensure all the shared objects we need at runtime are in the wheel
    add_target_libs_to_wheel(LIB_DIR cudf TARGETS arrow_shared nvcomp::nvcomp ${_nvcomp_link_libs})
  endif()
  # Since there are multiple subpackages of cudf._lib that require access to libcudf, we place the
  # library in the cudf directory as a single source of truth and modify the other rpaths
  # appropriately.
  set(cython_lib_dir cudf)
  install(TARGETS cudf DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(cudf/_lib)
add_subdirectory(udf_cpp)

include(cmake/Modules/ProtobufHelpers.cmake)
codegen_protoc(cudf/utils/metadata/orc_column_statistics.proto)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cudf PATHS "${cython_lib_dir}")
endif()
