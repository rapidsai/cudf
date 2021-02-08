# =============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# Build the list of supported architectures

set(SUPPORTED_CUDA_ARCHITECTURES "60" "62" "70" "72" "75" "80")

# Check for embedded vs workstation architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # This is being built for Linux4Tegra or SBSA ARM64
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "60" "70")
else()
  # This is being built for an x86 or x86_64 architecture
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "62" "72")
endif()

# CMake < 3.20 has a bug in FindCUDAToolkit where it won't
# properly detect the CUDAToolkit version when
# find_package(CUDAToolkit) occurs before enable_language(CUDA)
if(NOT DEFINED CUDAToolkit_VERSION AND CMAKE_CUDA_COMPILER)
  execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--version" OUTPUT_VARIABLE NVCC_OUT)
  if(NVCC_OUT MATCHES [=[ V([0-9]+)\.([0-9]+)\.([0-9]+)]=])
    set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(CUDAToolkit_VERSION  "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  endif()
  unset(NVCC_OUT)
endif()

if(CUDAToolkit_VERSION_MAJOR LESS 11)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "80")
endif()
if(CUDAToolkit_VERSION_MAJOR LESS 10)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "75")
endif()
if(CUDAToolkit_VERSION_MAJOR LESS 9)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "70")
endif()

# If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
# `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${SUPPORTED_CUDA_ARCHITECTURES})
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  unset(CMAKE_CUDA_ARCHITECTURES)
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
  include(${CUDF_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(CMAKE_CUDA_ARCHITECTURES)
endif()
