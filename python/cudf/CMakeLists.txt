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

if(POLICY CMP0094)
  # Make FindPython{,2,3} modules use LOCATION for lookup strategy.
  # https://cmake.org/cmake/help/latest/policy/CMP0094.html#policy:CMP0094
  cmake_policy(SET CMP0094 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0094 NEW)
endif()

include(../../fetch_rapids.cmake)

include(rapids-cuda)
# TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
# languages for the C++ project even if this project does not require those languges.
rapids_cuda_init_architectures(cudf-python)

set(cudf_version 22.10.00)

project(
  cudf-python
  VERSION ${cudf_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
            # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
            # languages for the C++ project even if this project does not require those languges.
            CUDA
)

option(FIND_CUDF_CPP "Search for existing CUDF C++ installations before defaulting to local files"
       OFF
)
option(CUDF_BUILD_WHEELS "Whether we're building a wheel for pypi" OFF)

# If the user requested it we attempt to find CUDF.
if(FIND_CUDF_CPP)
  find_package(cudf ${cudf_version} REQUIRED)
else()

  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  # Statically link dependencies if building wheels
  set(CPM_DOWNLOAD_rmm ${CUDF_BUILD_WHEELS})
  set(CPM_DOWNLOAD_spdlog ${CUDF_BUILD_WHEELS})
  set(CUDA_STATIC_RUNTIME ${CUDF_BUILD_WHEELS})

  # Need to set this so all the nvcomp targets are global, not only nvcomp::nvcomp
  # https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_PACKAGE_TARGETS_GLOBAL.html#variable:CMAKE_FIND_PACKAGE_TARGETS_GLOBAL
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ${CUDF_BUILD_WHEELS})

  set(_exclude_from_all "")
  if(CUDF_BUILD_WHEELS)
    # Don't install the cuDF C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp cudf-cpp ${_exclude_from_all})

  if(CUDF_BUILD_WHEELS)
    include(cmake/Modules/WheelHelpers.cmake)
    get_target_property(_nvcomp_link_libs nvcomp::nvcomp INTERFACE_LINK_LIBRARIES)
    # Ensure all the shared objects we need at runtime are in the wheel
    add_target_libs_to_wheel(
                     LIB_DIR cudf/_lib
                     TARGETS arrow_shared
                             nvcomp::nvcomp
                             ${_nvcomp_link_libs}
                             )
  endif()
  # Since there are multiple subpackages of cudf._lib that require access to libcudf, we place the
  # library in the _lib/cpp directory as a single source of truth and modify the other rpaths
  # appropriately.
  install(TARGETS cudf DESTINATION cudf/_lib)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(cudf/_lib)
