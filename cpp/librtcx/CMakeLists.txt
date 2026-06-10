# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(cmake/rapids_config.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  rapids_cpm_init()
endif()

project(
  rtcx
  VERSION 0.1.0
  LANGUAGES CXX
)

option(RTCX_STATIC_LINK_NVRTC "Use static linking for NVRTC" OFF)
option(RTCX_STATIC_LINK_NVJITLINK "Use static linking for nvJitLink" OFF)

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET zstd)
  set(CPM_DOWNLOAD_zstd ON)
  rapids_cpm_find(
    zstd 1.5.7
    GLOBAL_TARGETS zstd
    CPM_ARGS
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.7
    GIT_SHALLOW FALSE SOURCE_SUBDIR build/cmake
    OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
            "ZSTD_BUILD_PROGRAMS OFF" "BUILD_SHARED_LIBS OFF"
  )

  if(zstd_ADDED)
    # disable weak symbols support to hide tracing APIs as well
    target_compile_definitions(libzstd_static PRIVATE ZSTD_HAVE_WEAK_SYMBOLS=0)
    # expose experimental API
    target_compile_definitions(libzstd_static PUBLIC ZSTD_STATIC_LINKING_ONLY=0N)
    # suppress warnings from uninitialized variables and redefining ZSTD_STATIC_LINKING_ONLY
    target_compile_options(libzstd_static PRIVATE -w)
    add_library(zstd ALIAS libzstd_static)
  endif()
endif()

if(NOT TARGET nvtx3::nvtx3-cpp)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)
  rapids_cpm_nvtx3()
endif()

add_library(rtcx STATIC rtcx.cpp)
add_library(rtcx::rtcx ALIAS rtcx)

set_target_properties(
  rtcx
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED YES
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES
             POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  rtcx
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(rtcx PRIVATE zstd ${CMAKE_DL_LIBS} nvtx3::nvtx3-cpp)

if(RTCX_STATIC_LINK_NVRTC)
  target_link_libraries(rtcx PRIVATE CUDA::nvrtc_static CUDA::nvrtc_builtins_static)
  target_compile_definitions(rtcx PRIVATE RTCX_STATIC_LINK_LIBNVRTC=1)
else()
  target_compile_definitions(rtcx PRIVATE RTCX_STATIC_LINK_LIBNVRTC=0)
endif()

if(RTCX_STATIC_LINK_NVJITLINK)
  target_link_libraries(rtcx PRIVATE CUDA::nvJitLink_static)
  target_compile_definitions(rtcx PRIVATE RTCX_STATIC_LINK_LIBNVJITLINK=1)
else()
  target_compile_definitions(rtcx PRIVATE RTCX_STATIC_LINK_LIBNVJITLINK=0)
endif()
