#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

if(NOT CMAKE_CUDA_COMPILER)
    message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUDF_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
    if(CUDF_BUILD_TESTS OR CUDF_BUILD_BENCHMARKS)
        # Suppress parentheses warning which causes gmock to fail
        list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wno-parentheses)
    endif()
endif(CMAKE_COMPILER_IS_GNUCXX)

# Find the CUDAToolkit
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_VERSION_MAJOR: ${CUDAToolkit_VERSION_MAJOR}")
message(STATUS "CUDAToolkit_VERSION_MINOR: ${CUDAToolkit_VERSION_MINOR}")

# Auto-detect available GPU compute architectures
set(GPU_ARCHS "ALL" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for. Pass 'ALL' if you want to compile for all supported GPU architectures. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
    set(AUTO_DETECT_CUDA_ARCHITECTURES ON)
endif()

if(AUTO_DETECT_CUDA_ARCHITECTURES)
    set(GPU_ARCHS "")
else()
    set(GPU_ARCHS "ALL")
endif()

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
    # Check for embedded vs workstation architectures
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        # This is being built for Linux4Tegra or SBSA ARM64
        set(GPU_ARCHS "62")
        if((CUDAToolkit_VERSION_MAJOR EQUAL 9) OR (CUDAToolkit_VERSION_MAJOR GREATER 9))
            list(APPEND GPU_ARCHS "72")
        endif()
        if((CUDAToolkit_VERSION_MAJOR EQUAL 11) OR (CUDAToolkit_VERSION_MAJOR GREATER 11))
            list(APPEND GPU_ARCHS "75;80")
        endif()
    else()
        # This is being built for an x86 or x86_64 architecture
        set(GPU_ARCHS "60")
        if((CUDAToolkit_VERSION_MAJOR EQUAL 9) OR (CUDAToolkit_VERSION_MAJOR GREATER 9))
            list(APPEND GPU_ARCHS "70")
        endif()
        if((CUDAToolkit_VERSION_MAJOR EQUAL 10) OR (CUDAToolkit_VERSION_MAJOR GREATER 10))
            list(APPEND GPU_ARCHS "75")
        endif()
        if((CUDAToolkit_VERSION_MAJOR EQUAL 11) OR (CUDAToolkit_VERSION_MAJOR GREATER 11))
            list(APPEND GPU_ARCHS "80")
        endif()
    endif()
endif()

message("GPU_ARCHS = ${GPU_ARCHS}")

set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHS})

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
list(APPEND CUDF_CUDA_FLAGS -Werror=cross-execution-space-call)
list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

if(DISABLE_DEPRECATION_WARNING)
    list(APPEND CUDF_CXX_FLAGS -Wno-deprecated-declarations)
    list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
if(CMAKE_CUDA_LINEINFO)
    list(APPEND CUDF_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUDF_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()
