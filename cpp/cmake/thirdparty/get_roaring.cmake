# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

# This function finds CRoaring and sets up the necessary targets and include directories.
function(find_and_configure_roaring VERSION)
  # Download and build CRoaring via CPM
  CPMAddPackage(
    NAME roaring ${VERSION}
    GLOBAL_TARGETS roaring::roaring roaring-headers roaring-headers-cpp
    CPM_ARGS
    GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW FALSE
    SOURCE_SUBDIR build/cmake
    OPTIONS 
      "ROARING_BUILD_STATIC ON" "BUILD_SHARED_LIBS OFF"
  )

  if(roaring_ADDED)
    add_library(roaring INTERFACE IMPORTED)
    add_library(roaring-headers INTERFACE IMPORTED)
    add_library(roaring-headers-cpp INTERFACE IMPORTED)
  endif()

  if(DEFINED roaring_SOURCE_DIR)
    set(roaring_CPP_INCLUDE_DIR
        "${roaring_SOURCE_DIR}/cpp"
        PARENT_SCOPE
    )
    set(roaring_C_INCLUDE_DIR
    "${roaring_SOURCE_DIR}/include"
    PARENT_SCOPE
    )
  endif()

  # Export the find package root for downstream consumers
  message(STATUS "roaring_BINARY_DIR: ${roaring_BINARY_DIR}")
  rapids_export_find_package_root(BUILD roaring "${roaring_BINARY_DIR}" EXPORT_SET cudf-exports)
endfunction()

set(roaring_VERSION_cudf "4.3.5")
find_and_configure_roaring(${roaring_VERSION_cudf}) 