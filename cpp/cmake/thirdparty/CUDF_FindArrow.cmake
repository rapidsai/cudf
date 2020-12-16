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

set(CUDF_VERSION_Arrow 1.0.1)

set(ARROW_BUILD_SHARED ON)
set(ARROW_BUILD_STATIC OFF)

if(ARROW_STATIC_LIB)
    set(ARROW_BUILD_STATIC ON)
    set(ARROW_BUILD_SHARED OFF)
endif()

CPMFindPackage(NAME Arrow
    VERSION         ${CUDF_VERSION_Arrow}
    GIT_REPOSITORY  https://github.com/apache/arrow.git
    GIT_TAG         apache-arrow-${CUDF_VERSION_Arrow}
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   cpp
    OPTIONS         "CMAKE_VERBOSE_MAKEFILE         ON"
                    "ARROW_IPC                      ON"
                    "ARROW_CUDA                     ON"
                    "ARROW_DATASET                  ON"
                    "ARROW_WITH_BACKTRACE           ON"
                    "ARROW_BUILD_STATIC             ${ARROW_BUILD_STATIC}"
                    "ARROW_BUILD_SHARED             ${ARROW_BUILD_SHARED}"
                    "ARROW_DEPENDENCY_USE_SHARED    ${ARROW_BUILD_SHARED}"
                    "ARROW_JEMALLOC                 OFF"
)

message(STATUS "Arrow_DIR: ${Arrow_DIR}")
message(STATUS "Arrow_ADDED: ${Arrow_ADDED}")
message(STATUS "ArrowCUDA_DIR: ${ArrowCUDA_DIR}")
message(STATUS "ArrowCUDA_ADDED: ${ArrowCUDA_ADDED}")

if(NOT Arrow_ADDED)
    if(ArrowCUDA_DIR)
        set(Arrow_DIR "${ArrowCUDA_DIR}")
    elseif(Arrow_DIR)
        set(ArrowCUDA_DIR "${Arrow_DIR}")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${Arrow_DIR}")
endif()

if(NOT Arrow_ADDED AND (Arrow_DIR OR ArrowCUDA_DIR))

    find_package(Arrow)
    message(STATUS "Arrow_FOUND: ${Arrow_FOUND}")
    if(NOT Arrow_FOUND)
        message(FATAL_ERROR "Arrow package not found")
    endif()

    find_package(ArrowCUDA)
    message(STATUS "ArrowCUDA_FOUND: ${ArrowCUDA_FOUND}")
    if(NOT ArrowCUDA_FOUND)
        message(FATAL_ERROR "Arrow package not found")
    endif()

    if(ARROW_BUILD_STATIC)
        set(ARROW_LINKAGE STATIC)
        set(ARROW_LIBRARY ${ARROW_STATIC_LIB})
        set(ARROW_CUDA_LIBRARY ${ARROW_CUDA_STATIC_LIB})
    else()
        set(ARROW_LINKAGE SHARED)
        set(ARROW_LIBRARY ${ARROW_SHARED_LIB})
        set(ARROW_CUDA_LIBRARY ${ARROW_CUDA_SHARED_LIB})
    endif()

    add_library(arrow ${ARROW_LINKAGE} IMPORTED)
    add_library(arrow_cuda ${ARROW_LINKAGE} IMPORTED)
    target_include_directories(arrow INTERFACE ${ARROW_INCLUDE_DIR})
    target_include_directories(arrow_cuda INTERFACE ${ARROW_CUDA_INCLUDE_DIR})
    set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIBRARY})
    set_target_properties(arrow_cuda PROPERTIES IMPORTED_LOCATION ${ARROW_CUDA_LIBRARY})
endif()

if(NOT (Arrow_FOUND OR Arrow_ADDED))
    message(FATAL_ERROR "Arrow package not found")
endif()

message(STATUS "Arrow_SOURCE_DIR: ${Arrow_SOURCE_DIR}")
message(STATUS "Arrow_BINARY_DIR: ${Arrow_BINARY_DIR}")
