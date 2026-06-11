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
  include(rapids-export)
  include(rapids-find)
  rapids_cpm_init()
endif()

project(
  rtcx
  VERSION 0.1.0
  LANGUAGES CXX
)

option(RTCX_STATIC_LINK_NVRTC "Use static linking for NVRTC" OFF)
option(RTCX_STATIC_LINK_NVJITLINK "Use static linking for nvJitLink" OFF)

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET rtcx-exports
  INSTALL_EXPORT_SET rtcx-exports
)

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

# =============================================================================
# Install / Export
# =============================================================================
include(GNUInstallDirs)
include(${rapids-cmake-dir}/cmake/install_lib_dir.cmake)
rapids_cmake_install_lib_dir(lib_dir)

# Option to control install (default ON when built standalone, OFF when used as subdirectory)
option(RTCX_INSTALL "Enable installation of rtcx targets" ${PROJECT_IS_TOP_LEVEL})

set(rtcx_install_code_string
    [=[
# Embed functions (add_embed, embed_includes, embed_blob, embed)
# are included automatically so consumers don't need explicit include().
include("${CMAKE_CURRENT_LIST_DIR}/embed.cmake")

# Set rtcx_LIBCXX_DIR for consumers using embed_includes with libcxx headers.
set(rtcx_LIBCXX_DIR "${PACKAGE_PREFIX_DIR}/share/rtcx/libcxx")
]=]
)

string(
  CONFIGURE
    [=[
# Embed functions (add_embed, embed_includes, embed_blob, embed)
# are included automatically so consumers don't need explicit include().
include("@CMAKE_CURRENT_SOURCE_DIR@/embed.cmake")

# Set rtcx_LIBCXX_DIR for consumers using embed_includes with libcxx headers.
set(rtcx_LIBCXX_DIR "@CMAKE_CURRENT_SOURCE_DIR@/libcxx")
]=]
    rtcx_build_code_string
  @ONLY
)

if(NOT RTCX_INSTALL)
  set(rtcx_exclude_from_install EXCLUDE_FROM_ALL)
endif()

install(
  TARGETS rtcx
  EXPORT rtcx-exports
  ARCHIVE DESTINATION ${lib_dir}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ${rtcx_exclude_from_install}
)

if(RTCX_INSTALL)
  install(FILES rtcx.hpp sha256.hpp embed.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rtcx)
  install(FILES embed.cmake embed.in.cpp DESTINATION ${lib_dir}/cmake/rtcx)
  install(DIRECTORY libcxx/ DESTINATION ${CMAKE_INSTALL_DATADIR}/rtcx/libcxx)

  rapids_export(
    INSTALL rtcx
    EXPORT_SET rtcx-exports
    GLOBAL_TARGETS rtcx
    NAMESPACE rtcx::
    FINAL_CODE_BLOCK rtcx_install_code_string
  )
endif()

# Build-tree export (always, so CPM consumers can find_package from the build tree)
rapids_export(
  BUILD rtcx
  EXPORT_SET rtcx-exports
  GLOBAL_TARGETS rtcx
  NAMESPACE rtcx::
  FINAL_CODE_BLOCK rtcx_build_code_string
)
