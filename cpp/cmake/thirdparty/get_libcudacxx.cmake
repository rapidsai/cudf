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

# This function finds thrust and sets any additional necessary environment variables.
function(find_and_configure_libcudacxx)
  # Make sure we install libcudacxx beside our patched version of thrust
  include(GNUInstallDirs)
  set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/libcudf")
  set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_INCLUDEDIR}/lib")

  include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
  rapids_cpm_libcudacxx(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)

  # Store where CMake can find our custom Thrust install
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    INSTALL libcudacxx
    [=[${CMAKE_CURRENT_LIST_DIR}/../../../include/libcudf/lib/rapids/cmake/libcudacxx]=]
    EXPORT_SET cudf-exports
    CONDITION libcudacxx_SOURCE_DIR
  )
endfunction()

find_and_configure_libcudacxx()
