# =============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  rapids_cpm_package_override("${CMAKE_CURRENT_LIST_DIR}/patches/cccl_override.json")

  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
endfunction()

find_and_configure_cccl()
