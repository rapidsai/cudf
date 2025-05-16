# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
  include(${rapids-cmake-dir}/cpm/cuco.cmake)

  rapids_cpm_cuco(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
endfunction()

find_and_configure_cucollections()
