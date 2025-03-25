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

# Use CPM to find or clone libzstd
function(find_and_configure_zstd)
  include("${rapids-cmake-dir}/cpm/package_override.cmake")
  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/zstd_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(zstd version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(zstd ${version} patch_command build_patch_only)

  set(CPM_DOWNLOAD_zstd ON)
  rapids_cpm_find(
    zstd ${version} ${build_patch_only}
    GLOBAL_TARGETS zstd
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow} SOURCE_SUBDIR build/cmake
    OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
            "ZSTD_BUILD_PROGRAMS OFF"
  )

  if(zstd_ADDED)
    # disable weak symbols support to hide tracing APIs as well
    target_compile_definitions(libzstd_static PRIVATE ZSTD_HAVE_WEAK_SYMBOLS=0)
    # expose experimental API
    target_compile_definitions(libzstd_static PUBLIC ZSTD_STATIC_LINKING_ONLY=0N)
    # suppress warnings from uninitialized variables and redefining ZSTD_STATIC_LINKING_ONLY
    target_compile_options(libzstd_static PRIVATE -w)
    add_library(zstd ALIAS libzstd_static)
  endif()

  if(DEFINED zstd_SOURCE_DIR)
    set(ZSTD_INCLUDE_DIR
        "${zstd_SOURCE_DIR}/lib"
        PARENT_SCOPE
    )
  endif()
  rapids_export_find_package_root(BUILD zstd "${zstd_BINARY_DIR}" EXPORT_SET cudf-exports)

endfunction()

find_and_configure_zstd()
