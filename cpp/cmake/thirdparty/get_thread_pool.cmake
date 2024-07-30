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

# Need to call rapids_cpm_bs_thread_pool to get support for an installed version of thread-pool and
# to support installing it ourselves
function(find_and_configure_thread_pool)
  include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

  # Find or install thread-pool
  rapids_cpm_bs_thread_pool()

endfunction()

find_and_configure_thread_pool()
