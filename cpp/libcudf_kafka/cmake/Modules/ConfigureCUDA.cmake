#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

# Find the CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Must come after find_package(CUDAToolkit) because we symlink
# ccache as a compiler front-end for nvcc in gpuCI CPU builds.
enable_language(CUDA)

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
endif()

message(VERBOSE "CUDF: CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(VERBOSE "CUDF: CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDF: CUDA_VERSION: ${CUDA_VERSION}")

# Auto-detect available GPU compute architectures

include(${CMAKE_SOURCE_DIR}/cmake/Modules/SetGPUArchs.cmake)
message(STATUS "CUDF: Building CUDF for GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUDF_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
    if(CUDF_BUILD_TESTS OR CUDF_BUILD_BENCHMARKS)
        # Suppress parentheses warning which causes gmock to fail
        list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wno-parentheses)
    endif()
endif()

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
list(APPEND CUDF_CUDA_FLAGS -Werror=cross-execution-space-call)
list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

if(DISABLE_DEPRECATION_WARNING)
    list(APPEND CUDF_CXX_FLAGS -Wno-deprecated-declarations)
    list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
if(CUDA_ENABLE_LINEINFO)
    list(APPEND CUDF_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(VERBOSE "CUDF: Building with debugging flags")
    list(APPEND CUDF_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()