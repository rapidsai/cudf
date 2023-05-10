# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
function(find_and_configure_cufile)

  list(APPEND CMAKE_MODULE_PATH ${CUDF_SOURCE_DIR}/cmake/Modules)
  rapids_find_package(cuFile QUIET)

  if(cuFile_FOUND AND NOT BUILD_SHARED_LIBS)
    include("${rapids-cmake-dir}/export/find_package_file.cmake")
    rapids_export_find_package_file(
      BUILD "${CUDF_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" cudf-exports
    )
    rapids_export_find_package_file(
      INSTALL "${CUDF_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" cudf-exports
    )
  endif()
endfunction()

find_and_configure_cufile()
