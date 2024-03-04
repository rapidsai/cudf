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

# This function finds NVTX and sets any additional necessary environment variables.
function(find_and_configure_nvtx)
  rapids_cpm_find(
    NVTX3 3.1.0
    GLOBAL_TARGETS nvtx3-c nvtx3-cpp
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE SOURCE_SUBDIR c
  )
endfunction()

find_and_configure_nvtx()
