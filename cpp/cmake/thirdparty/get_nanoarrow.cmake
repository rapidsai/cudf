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

# This function finds nanoarrow and sets any additional necessary environment variables.
function(find_and_configure_nanoarrow)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Only run if PKG_VERSION is < 0.5.0
  if(PKG_VERSION VERSION_LESS 0.5.0)
    set(patch_files_to_run "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/nanoarrow_cmake.diff")
    set(patch_issues_to_ref
        "Fix issues with nanoarrow CMake [https://github.com/apache/arrow-nanoarrow/pull/406]"
    )
    set(patch_script "${CMAKE_BINARY_DIR}/rapids-cmake/patches/nanoarrow/patch.cmake")
    set(log_file "${CMAKE_BINARY_DIR}/rapids-cmake/patches/nanoarrow/log")
    string(TIMESTAMP current_year "%Y" UTC)
    configure_file(
      ${rapids-cmake-dir}/cpm/patches/command_template.cmake.in "${patch_script}" @ONLY
    )
  else()
    message(
      FATAL_ERROR
        "Nanoarrow version ${PKG_VERSION} already contains the necessary patch. Please remove this patch from cudf."
    )
  endif()

  rapids_cpm_find(
    nanoarrow ${PKG_VERSION}
    GLOBAL_TARGETS nanoarrow
    CPM_ARGS
    GIT_REPOSITORY https://github.com/${PKG_FORK}/arrow-nanoarrow.git
    GIT_TAG ${PKG_PINNED_TAG}
    # TODO: Commit hashes are not supported with shallow clones. Can switch this if and when we pin
    # to an actual tag.
    GIT_SHALLOW FALSE
    PATCH_COMMAND ${CMAKE_COMMAND} -P ${patch_script}
    OPTIONS "BUILD_SHARED_LIBS OFF" "NANOARROW_NAMESPACE cudf"
  )
  set_target_properties(nanoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
  rapids_export_find_package_root(BUILD nanoarrow "${nanoarrow_BINARY_DIR}" EXPORT_SET cudf-exports)
endfunction()

find_and_configure_nanoarrow(
  VERSION 0.4.0 FORK apache PINNED_TAG c97720003ff863b81805bcdb9f7c91306ab6b6a8
)
