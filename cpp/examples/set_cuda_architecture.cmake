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

# get the CUDF_TAG from fetch_dependencies.cmake (remove all whitespace & parse line: set(CUDF_TAG
# ..))
execute_process(
  COMMAND
    bash -c
    "sed '/^$/d;s/[[:blank:]]//g' ./fetch_dependencies.cmake | grep 'set(CUDF_TAG' | sed 's/.*CUDF_TAG//' | sed 's/.$//'"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CUDF_TAG
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/libcudf_cpp_examples_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/${CUDF_TAG}/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/libcudf_cpp_examples_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/libcudf_cpp_examples_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
