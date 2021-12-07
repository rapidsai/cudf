# =============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
function(find_and_configure_nvcomp VERSION)

  # Find or install nvcomp
  rapids_cpm_find(
    nvcomp ${VERSION}
    GLOBAL_TARGETS nvcomp::nvcomp
    CPM_ARGS GITHUB_REPOSITORY NVIDIA/nvcomp
    GIT_TAG c435afaf4ba8a8d12f379d688effcb185886cec1
    OPTIONS "BUILD_STATIC ON" "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
  )

  if(NOT TARGET nvcomp::nvcomp)
    add_library(nvcomp::nvcomp ALIAS nvcomp)
  endif()

  # Per-thread default stream
  if(TARGET nvcomp AND PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif()

endfunction()

set(CUDF_MIN_VERSION_nvCOMP 2.1.0)

find_and_configure_nvcomp(${CUDF_MIN_VERSION_nvCOMP})
