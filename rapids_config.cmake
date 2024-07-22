# =============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _rapids_version)
if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])]])
  set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
  set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
  set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
else()
  string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
  message(
    FATAL_ERROR
      "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}"
  )
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake")
  set(rapids-cmake-repo KyleFromNVIDIA/rapids-cmake)
  set(rapids-cmake-branch bs-thread-pool)
  file(
    DOWNLOAD
    "https://raw.githubusercontent.com/KyleFromNVIDIA/rapids-cmake/bs-thread-pool/RAPIDS.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake"
  )
endif()
include("${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS-${RAPIDS_VERSION_MAJOR_MINOR}.cmake")
