# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

# This function finds cccl and sets any additional necessary environment variables.
function(find_and_configure_cccl)

  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  set(cudf_patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches")
  rapids_cpm_package_override("${cudf_patch_dir}/cccl_override.json")

  # Make sure we install cccl into the `include/libcudf` subdirectory instead of the default
  include(GNUInstallDirs)
  set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/libcudf")
  set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_INCLUDEDIR}/lib")

  # Store where CMake can find our custom CCCL install
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    INSTALL CCCL [=[${CMAKE_CURRENT_LIST_DIR}/../../../include/libcudf/lib/rapids/cmake/cccl]=]
    EXPORT_SET cudf-exports
  )

  # Find or install CCCL with our custom set of patches
  rapids_cpm_cccl(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)

endfunction()

find_and_configure_cccl()
