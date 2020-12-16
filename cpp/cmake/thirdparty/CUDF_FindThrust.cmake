#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

set(CUDF_MIN_VERSION_Thrust 1.10.0)

CPMFindPackage(NAME Thrust
    VERSION         ${CUDF_MIN_VERSION_Thrust}
    GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
    GIT_TAG         ${CUDF_MIN_VERSION_Thrust}
    GIT_SHALLOW     TRUE
    FORCE           TRUE
    # If there is no pre-installed thrust we can use, we'll install our fetched copy together with cuDF
    OPTIONS         "THRUST_INSTALL TRUE"
    PATCH_COMMAND   patch -p1 -N < ${CUDA_DATAFRAME_SOURCE_DIR}/cmake/thrust.patch || true)

message(STATUS "Thrust_ADDED: ${Thrust_ADDED}")
message(STATUS "Thrust_FOUND: ${Thrust_FOUND}")

if(NOT (Thrust_ADDED OR Thrust_FOUND))
    message(FATAL_ERROR "Thrust package not found")
endif()

thrust_create_target(cudf::Thrust FROM_OPTIONS)

message(STATUS "Thrust_SOURCE_DIR: ${Thrust_SOURCE_DIR}")
message(STATUS "Thrust_BINARY_DIR: ${Thrust_BINARY_DIR}")
