# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

include(${CMAKE_CURRENT_LIST_DIR}/versions.cmake)

set(CPM_DOWNLOAD_VERSION v0.38.5)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

# find or build it via CPM
CPMFindPackage(
  NAME cudf
  FIND_PACKAGE_ARGUMENTS "PATHS ${cudf_ROOT} ${cudf_ROOT}/latest" GIT_REPOSITORY
                         https://github.com/kingcrimsontianyu/cudf
  GIT_TAG "orc-debug"
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)
