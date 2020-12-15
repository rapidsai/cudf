# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# Use ccache if possible.
option(USE_CCACHE_COMPILER_LAUNCHER "Use ccache to launch C/C++/CUDA compilers" ON)
set(CCACHE_CONFIGPATH "OFF" CACHE FILEPATH "Location of the ccache configuration file")

if(USE_CCACHE_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    if(CCACHE_CONFIGPATH)
      set(CCACHE_PROGRAM CCACHE_CONFIGPATH=${CCACHE_CONFIGPATH} ${CCACHE_PROGRAM})
    endif()
    if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER)
      message(STATUS "Using C compiler launcher: ${CCACHE_PROGRAM}")
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
      message(STATUS "Using CXX compiler launcher: ${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
      message(STATUS "Using CUDA compiler launcher: ${CCACHE_PROGRAM}")
      set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
  endif()
endif()
