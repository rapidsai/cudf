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

  set(CPM_DOWNLOAD_zstd ON)
  rapids_cpm_find(
    zstd 1.5.7
    GLOBAL_TARGETS zstd
    CPM_ARGS
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.7
    GIT_SHALLOW FALSE SOURCE_SUBDIR build/cmake
    OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
            "ZSTD_BUILD_PROGRAMS OFF"
  )

  # we need this to disable weak symbols support to hide tracing APIs as well
  if(zstd_ADDED)
    target_compile_definitions(libzstd_static PRIVATE ZSTD_HAVE_WEAK_SYMBOLS=0)
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
