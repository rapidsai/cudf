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

set(NVCOMP_ROOT "${CMAKE_BINARY_DIR}/nvcomp")

set(NVCOMP_CMAKE_ARGS "-DUSE_RMM=ON -DCUB_DIR=${CUB_INCLUDE}")

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/Nvcomp.CMakeLists.txt.cmake"
               "${NVCOMP_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${NVCOMP_ROOT}/build")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE NVCOMP_CONFIG
                WORKING_DIRECTORY ${NVCOMP_ROOT})

if(NVCOMP_CONFIG)
    message(FATAL_ERROR "Configuring nvcomp failed: " ${NVCOMP_CONFIG})
endif(NVCOMP_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "NVCOMP BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "NVCOMP BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "NVCOMP BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
                RESULT_VARIABLE NVCOMP_BUILD
                WORKING_DIRECTORY ${NVCOMP_ROOT}/build)

if(NVCOMP_BUILD)
    message(FATAL_ERROR "Building nvcomp failed: " ${NVCOMP_BUILD})
endif(NVCOMP_BUILD)

message(STATUS "nvcomp build completed at: " ${NVCOMP_ROOT}/build)

set(NVCOMP_INCLUDE_DIR "${NVCOMP_ROOT}/build/include")
set(NVCOMP_LIBRARY_DIR "${NVCOMP_ROOT}/build/lib")

find_library(NVCOMP_LIB nvcomp
    NO_DEFAULT_PATH
    HINTS "${NVCOMP_LIBRARY_DIR}")

if(NVCOMP_LIB)
    message(STATUS "nvcomp library: " ${NVCOMP_LIB})
    set(NVCOMP_FOUND TRUE)
endif(NVCOMP_LIB)
