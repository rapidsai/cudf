#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_cuco)
  if(CUCO_INCLUDE)
    set(CUCO_INCLUDE_DIR "${CUCO_INCLUDE}" PARENT_SCOPE)
    return()
  endif()
  if(CUCO_INCLUDE_DIR)
    set(CUCO_INCLUDE_DIR ${CUCO_INCLUDE_DIR} PARENT_SCOPE)
    return()
  endif()
  CPMFindPackage(NAME cuco
    GITHUB_REPOSITORY PointKernel/cuCollections
    GIT_TAG           static-multi-map
    GIT_SHALLOW       TRUE
    DOWNLOAD_ONLY     TRUE
    OPTIONS           "BUILD_BENCHMARKS OFF"
                      "BUILD_EXAMPLES OFF"
                      "BUILD_TESTS OFF")
    set(CUCO_INCLUDE_DIR "${cuco_SOURCE_DIR}/include" PARENT_SCOPE)
endfunction()

find_and_configure_cuco()
