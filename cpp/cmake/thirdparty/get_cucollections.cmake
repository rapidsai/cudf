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

# This function finds cuCollections and performs any additional configuration.
function(find_and_configure_cucollections)
  rapids_cpm_find(
    cuco 0.0.1 ￼
    GLOBAL_TARGETS cuco::cuco ￼
    CPM_ARGS ￼
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git ￼
    GIT_TAG 31e1d5df6869ef6cb60f36a614b30a244cf3bd78 ￼
    OPTIONS "BUILD_TESTS OFF" ￼ "BUILD_BENCHMARKS OFF" ￼ "BUILD_EXAMPLES OFF" ￼
  )
endfunction()

find_and_configure_cucollections()
