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

# This function finds rmm and sets any additional necessary environment variables.
function(find_and_configure_thread_pool)
  rapids_cpm_find(
    BS_thread_pool 4.1.0
    CPM_ARGS
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG 097aa718f25d44315cadb80b407144ad455ee4f9
    GIT_SHALLOW TRUE
  )
  add_library(BS_thread_pool INTERFACE)
  target_include_directories(BS_thread_pool INTERFACE ${BS_thread_pool_SOURCE_DIR}/include)
  target_compile_definitions(BS_thread_pool INTERFACE "BS_THREAD_POOL_ENABLE_PAUSE=1")
endfunction()

find_and_configure_thread_pool()
