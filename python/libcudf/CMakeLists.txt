# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(libcudf-python)
rapids_cpm_init()

project(
  libcudf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_NVCOMP_RUNTIME_WHEEL "Use the nvcomp wheel at runtime instead of the system library" OFF)

# Check if cudf is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cudf package.
find_package(cudf "${RAPIDS_VERSION}")

if(cudf_FOUND)
  return()
endif()

unset(cudf_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)

if(USE_NVCOMP_RUNTIME_WHEEL)
  set(CUDF_EXPORT_NVCOMP OFF)
endif()

include(../../cpp/cmake/thirdparty/get_nvcomp.cmake)
include(cmake/Modules/WheelHelpers.cmake)

# Install only the specific libnvcomp.so.4 library instead of all nvcomp targets
if(TARGET nvcomp::nvcomp)
  # Check if the nvcomp target is imported
  get_target_property(is_imported nvcomp::nvcomp IMPORTED)
  if(is_imported)
    message(WARNING "nvcomp target is imported")
    # Debug: List all properties to see what's available
    get_target_property(nvcomp_type nvcomp::nvcomp TYPE)
    message(WARNING "nvcomp target type: ${nvcomp_type}")

    # Try to get all location-related properties
    get_target_property(nvcomp_loc nvcomp::nvcomp IMPORTED_LOCATION)
    get_target_property(nvcomp_loc_release nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
    get_target_property(nvcomp_loc_debug nvcomp::nvcomp IMPORTED_LOCATION_DEBUG)
    get_target_property(nvcomp_loc_noconfig nvcomp::nvcomp IMPORTED_LOCATION_NOCONFIG)

    message(WARNING "IMPORTED_LOCATION: ${nvcomp_loc}")
    message(WARNING "IMPORTED_LOCATION_RELEASE: ${nvcomp_loc_release}")
    message(WARNING "IMPORTED_LOCATION_DEBUG: ${nvcomp_loc_debug}")
    message(WARNING "IMPORTED_LOCATION_NOCONFIG: ${nvcomp_loc_noconfig}")
  else()
    message(WARNING "nvcomp target is not imported")
  endif()
  # Get the library file path and install only libnvcomp.so.4 Try IMPORTED_LOCATION first (for
  # single-config generators)
  get_target_property(nvcomp_lib_path nvcomp::nvcomp IMPORTED_LOCATION)
  if(NOT nvcomp_lib_path)
    # For multi-config generators, try IMPORTED_LOCATION_RELEASE
    get_target_property(nvcomp_lib_path nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
  endif()
  if(NOT nvcomp_lib_path)
    # Fallback to IMPORTED_LOCATION_DEBUG
    get_target_property(nvcomp_lib_path nvcomp::nvcomp IMPORTED_LOCATION_DEBUG)
  endif()
  if(NOT nvcomp_lib_path)
    # Try to get the location using generator expressions
    get_target_property(nvcomp_lib_path nvcomp::nvcomp IMPORTED_LOCATION_NOCONFIG)
  endif()
  # Print the library path
  message(WARNING "nvcomp_lib_path: ${nvcomp_lib_path}")
  if(nvcomp_lib_path)
    # Extract just the filename to check if it's the right library
    get_filename_component(nvcomp_lib_name ${nvcomp_lib_path} NAME)
    # Print the library name
    message(WARNING "nvcomp_lib_name: ${nvcomp_lib_name}")
    if(nvcomp_lib_name MATCHES "libnvcomp\\.so\\.4")
      # Print the library path
      message(WARNING "Installing nvcomp_lib_path: ${nvcomp_lib_path}")
      install(FILES ${nvcomp_lib_path} DESTINATION ${SKBUILD_PLATLIB_DIR}/libcudf/lib64/)
    else()
      # If the main target doesn't point to .so.4, try to find it in the same directory
      get_filename_component(nvcomp_lib_dir ${nvcomp_lib_path} DIRECTORY)
      set(nvcomp_so4_path "${nvcomp_lib_dir}/libnvcomp.so.4")
      # Print the library path
      message(WARNING "nvcomp_so4_path: ${nvcomp_so4_path}")
      if(EXISTS ${nvcomp_so4_path})
        # Print the library path
        message(WARNING "Installing nvcomp_so4_path: ${nvcomp_so4_path}")
        install(FILES ${nvcomp_so4_path} DESTINATION ${SKBUILD_PLATLIB_DIR}/libcudf/lib64/)
      else()
        # Fallback to the original behavior if we can't find the specific library
        message(WARNING "Falling back to original behavior nvcomp_so4_path not found")
        install_aliased_imported_targets(
          TARGETS nvcomp::nvcomp DESTINATION ${SKBUILD_PLATLIB_DIR}/libcudf/lib64/
        )
      endif()
    endif()
  else()
    # Fallback to the original behavior if we can't get the library path
    message(WARNING "Falling back to original behavior nvcomp_lib_path not found")
    install_aliased_imported_targets(
      TARGETS nvcomp::nvcomp DESTINATION ${SKBUILD_PLATLIB_DIR}/libcudf/lib64/
    )
  endif()
endif()

set(CUDA_STATIC_RUNTIME ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(../../cpp cudf-cpp)

if(USE_NVCOMP_RUNTIME_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nvcomp")
  set_property(
    TARGET cudf
    PROPERTY INSTALL_RPATH ${rpaths}
    APPEND
  )
endif()
