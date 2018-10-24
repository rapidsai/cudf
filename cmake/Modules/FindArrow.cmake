#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
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

# - Find Apache Arrow (libarrow.a)
# ARROW_ROOT hints the location
#
# This module defines
# ARROW_FOUND
# ARROW_INCLUDEDIR Preferred include directory e.g. <prefix>/include
# ARROW_INCLUDE_DIR, directory containing Arrow headers
# ARROW_LIBS, directory containing Arrow libraries
# ARROW_STATIC_LIB, path to libarrow.a
# arrow - static library

# If ARROW_ROOT is not defined try to search in the default system path
if ("${ARROW_ROOT}" STREQUAL "")
    set(ARROW_ROOT "/usr")
endif()

set(ARROW_SEARCH_LIB_PATH
  ${ARROW_ROOT}/lib
  ${ARROW_ROOT}/lib/x86_64-linux-gnu
  ${ARROW_ROOT}/lib64
)

set(ARROW_SEARCH_INCLUDE_DIR
  ${ARROW_ROOT}/include/arrow
)

find_path(ARROW_INCLUDE_DIR api.h
    PATHS ${ARROW_SEARCH_INCLUDE_DIR}
    NO_DEFAULT_PATH
    DOC "Path to Apache Arrow headers"
)

find_library(ARROW_LIBS NAMES arrow
    PATHS ${ARROW_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH
    DOC "Path to Apache Arrow library"
)

find_library(ARROW_STATIC_LIB NAMES libarrow.a
    PATHS ${ARROW_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH
    DOC "Path to Apache Arrow static library"
)

if (NOT ARROW_LIBS OR NOT ARROW_STATIC_LIB)
    message(FATAL_ERROR "Apache Arrow includes and libraries NOT found. "
      "Looked for headers in ${ARROW_SEARCH_INCLUDE_DIR}, "
      "and for libs in ${ARROW_SEARCH_LIB_PATH}")
    set(ARROW_FOUND FALSE)
else()
    set(ARROW_INCLUDEDIR ${ARROW_ROOT}/include/)
    set(ARROW_FOUND TRUE)
    add_library(arrow STATIC IMPORTED)
    set_target_properties(arrow PROPERTIES IMPORTED_LOCATION "${ARROW_STATIC_LIB}")

    # Determine arrow version information for CPP macros
    get_filename_component(ARROW_STATIC_LIB_DIR ${ARROW_STATIC_LIB} DIRECTORY)
    file(STRINGS ${ARROW_STATIC_LIB_DIR}/pkgconfig/arrow.pc _ARROW_VERSION REGEX "^Version: ([0-9]+\\.[0-9]+\\.[0-9]+)")
    STRING(REGEX REPLACE "^Version: ([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" ARROW_VERSION_MAJOR "${_ARROW_VERSION}")
    STRING(REGEX REPLACE "^Version: [0-9]+\\.([0-9]+)\\.[0-9]+" "\\1" ARROW_VERSION_MINOR "${_ARROW_VERSION}")
    STRING(REGEX REPLACE "^Version: [0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" ARROW_VERSION_PATCH "${_ARROW_VERSION}")
    math(EXPR ARROW_NUMERIC_VERSION "(${ARROW_VERSION_MAJOR}+0) * 10000 + (${ARROW_VERSION_MINOR}+0) * 100 + (${ARROW_VERSION_PATCH}+0)")
    message(STATUS "ARROW_NUMERIC_VERSION=${ARROW_NUMERIC_VERSION}")
    ADD_DEFINITIONS(-DARROW_VERSION=${ARROW_NUMERIC_VERSION})

    # see arrow/ipc/message.h
    if (ARROW_NUMERIC_VERSION EQUAL 0)
      message(FATAL_ERROR "Apache Arrow version indetermined. Check for any error messages above.")
    elseif (ARROW_NUMERIC_VERSION LESS 200)
      ADD_DEFINITIONS(-DARROW_METADATA_V1)
    elseif (ARROW_NUMERIC_VERSION LESS 300)
      ADD_DEFINITIONS(-DARROW_METADATA_V2)
    elseif (ARROW_NUMERIC_VERSION LESS 800)
      ADD_DEFINITIONS(-DARROW_METADATA_V3)
    else()
      ADD_DEFINITIONS(-DARROW_METADATA_V4)
    endif()
endif ()

mark_as_advanced(
  ARROW_FOUND
  ARROW_INCLUDEDIR
  ARROW_INCLUDE_DIR
  ARROW_LIBS
  ARROW_STATIC_LIB
  arrow
)
