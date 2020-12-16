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

set(CUDF_MIN_VERSION_rmm "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")

set(rmm_FOUND NO)

if(DEFINED ENV{RMM_HOME} AND (DEFINED ENV{RMM_ROOT}))
    add_subdirectory("$ENV{RMM_HOME}" "$ENV{RMM_ROOT}" EXCLUDE_FROM_ALL)
    if(TARGET rmm::rmm)
        set(rmm_FOUND YES)
        set(rmm_SOURCE_DIR "$ENV{RMM_HOME}")
        set(rmm_BINARY_DIR "$ENV{RMM_ROOT}")
    endif()
endif()

if(NOT rmm_FOUND AND (DEFINED ENV{RMM_ROOT}))
    find_package(RMM ${CUDF_MIN_VERSION_rmm} CONFIG)
    set(rmm_FOUND ${RMM_FOUND})
endif()
    
if(NOT rmm_FOUND)
    CPMAddPackage(NAME      rmm
        VERSION             ${CUDF_MIN_VERSION_rmm}
        GIT_REPOSITORY      https://github.com/rapidsai/rmm.git
        GIT_TAG             branch-${CUDF_MIN_VERSION_rmm}
        GIT_SHALLOW         TRUE
        OPTIONS             "BUILD_TESTS OFF"
                            "BUILD_BENCHMARKS OFF"
                            "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                            "CUDAToolkit_ROOT_DIR=${CUDAToolkit_ROOT_DIR}"
                            "CUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}"
                            "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}"
    )
endif()

message(STATUS "rmm_ADDED: ${rmm_ADDED}")
message(STATUS "rmm_FOUND: ${rmm_FOUND}")

if(NOT (rmm_FOUND OR rmm_ADDED))
    message(FATAL_ERROR "RMM package not found")
endif()

message(STATUS "rmm_SOURCE_DIR: ${rmm_SOURCE_DIR}")
message(STATUS "rmm_BINARY_DIR: ${rmm_BINARY_DIR}")
