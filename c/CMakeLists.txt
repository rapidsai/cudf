# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  CUDF_C
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX C
)

set(CMAKE_INSTALL_MESSAGE LAZY)

option(BUILD_TESTS "Build cudf_c unit-tests" ON)
option(BUILD_SHARED_LIBS "Build cudf_c shared libraries" ON)

rapids_cmake_build_type(Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET cudf::cudf)
  find_package(cudf "${RAPIDS_VERSION}" QUIET)
  if(NOT cudf_FOUND AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../cpp/build/latest/cudf-config.cmake")
    set(cudf_DIR "${CMAKE_CURRENT_LIST_DIR}/../cpp/build/latest")
    find_package(cudf "${RAPIDS_VERSION}" REQUIRED)
  elseif(NOT cudf_FOUND)
    find_package(cudf "${RAPIDS_VERSION}" REQUIRED)
  endif()
endif()

add_library(
  cudf_c SHARED src/core/c_api.cpp src/column.cpp src/table.cpp src/binaryop.cpp src/join.cpp
                src/interop/arrow.cpp src/io/parquet.cpp
)
add_library(cudf::c_api ALIAS cudf_c)

set_target_properties(
  cudf_c
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             EXPORT_NAME c_api
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
)

if(PROJECT_IS_TOP_LEVEL)
  include(cmake/determine_cudf_abi_version.cmake)
  determine_cudf_abi_version(${RAPIDS_VERSION} MAJOR abi_major MINOR abi_minor)
  set_target_properties(cudf_c PROPERTIES VERSION "${abi_major}.${abi_minor}")
  set_target_properties(cudf_c PROPERTIES SOVERSION "${abi_major}")
endif()

target_include_directories(
  cudf_c
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include>"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../cpp/build/latest/_deps/nanoarrow-src/src"
          "${CMAKE_CURRENT_LIST_DIR}/../cpp/build/latest/_deps/nanoarrow-build/src"
)

target_link_libraries(cudf_c PRIVATE cudf::cudf)

if(PROJECT_IS_TOP_LEVEL)
  rapids_cmake_install_lib_dir(lib_dir)
  include(GNUInstallDirs)

  install(
    TARGETS cudf_c
    DESTINATION ${lib_dir}
    COMPONENT c_api
    EXPORT cudf-c-exports
  )
  install(
    DIRECTORY include/cudf
    COMPONENT c_api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
