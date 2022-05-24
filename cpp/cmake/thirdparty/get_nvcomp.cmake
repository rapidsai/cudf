# =============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# This function finds nvcomp and sets any additional necessary environment variables.
function(find_and_configure_nvcomp VERSION_MIN VERSION_MAX)
  # Search for latest version of nvComp
  rapids_find_package(nvcomp ${VERSION_MAX} QUIET)
  # If latest isn't found, fall back to building oldest support from source
  rapids_cpm_find(
    nvcomp ${VERSION_MIN}
    GLOBAL_TARGETS nvcomp::nvcomp
    CPM_ARGS GITHUB_REPOSITORY NVIDIA/nvcomp
    GIT_TAG v${VERSION_MIN}
    OPTIONS "BUILD_STATIC ON" "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
  )

  if(nvcomp_BINARY_DIR)
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD nvcomp "${nvcomp_BINARY_DIR}" cudf-exports)
  endif()

  if(NOT TARGET nvcomp::nvcomp)
    add_library(nvcomp::nvcomp ALIAS nvcomp)
  endif()

  # Per-thread default stream
  if(TARGET nvcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif()
endfunction()

set(CUDF_MIN_VERSION_nvCOMP 2.2.0)
set(CUDF_MAX_VERSION_nvCOMP 2.3.0)
find_and_configure_nvcomp(${CUDF_MIN_VERSION_nvCOMP} ${CUDF_MAX_VERSION_nvCOMP})
