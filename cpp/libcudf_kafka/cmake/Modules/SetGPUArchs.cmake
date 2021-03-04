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

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "80")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "75")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9)
  list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES "70")
endif()

if(CUDF_BUILD_FOR_DETECTED_ARCHS)
  include(${CMAKE_SOURCE_DIR}/cmake/Modules/EvalGPUArchs.cmake)
  evaluate_gpu_archs(CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    set(CUDF_BUILD_FOR_ALL_ARCHS TRUE)
  else()
    set(CUDF_BUILD_FOR_ALL_ARCHS FALSE)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
  endif()
endif()

if(CUDF_BUILD_FOR_ALL_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES ${SUPPORTED_CUDA_ARCHITECTURES})
  # CMake architecture list entry of "80" means to build compute and sm.
  # What we want is for the newest arch only to build that way
  # while the rest built only for sm.
  list(SORT CMAKE_CUDA_ARCHITECTURES ORDER ASCENDING)
  list(POP_BACK CMAKE_CUDA_ARCHITECTURES latest_arch)
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
  list(APPEND CMAKE_CUDA_ARCHITECTURES ${latest_arch})
endif()