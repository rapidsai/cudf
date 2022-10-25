# =============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/vyasr/rapids-cmake/feature/rapids_cython_lib_rpath/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake
  )
endif()

set(rapids-cmake-repo vyasr/rapids-cmake)
set(rapids-cmake-branch feature/rapids_cython_lib_rpath)

include(${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
