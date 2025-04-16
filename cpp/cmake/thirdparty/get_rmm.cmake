# =============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
function(find_and_configure_rmm)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)

  if(CUDF_BUILD_STACKTRACE_DEBUG)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)
    set(cudf_patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches")
    rapids_cpm_package_override("${cudf_patch_dir}/rmm_override.json")
  endif()

  # Find or install RMM
  rapids_cpm_rmm(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)

endfunction()

find_and_configure_rmm()
