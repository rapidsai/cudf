#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# - Find FlatBuffers
# FLATBUFFERS_HOME hints the install location (directory where you flatbuffers is installed)
#
# This module defines
# FLATBUFFERS_FOUND
# FLATBUFFERS_FLATC_EXECUTABLE
# FLATBUFFERS_INCLUDEDIR Preferred include directory e.g. <prefix>/include
# FLATBUFFERS_INCLUDE_DIR, directory containing flatbuffers headers
# FLATBUFFERS_LIBS, flatbuffers libraries
# FLATBUFFERS_LIBDIR, directory containing flatbuffers libraries
# FLATBUFFERS_STATIC_LIB, path to flatbuffers.a
# flatbuffers - static library

# If FLATBUFFERS_HOME is not defined try to search in the default system path
if ("${FLATBUFFERS_HOME}" STREQUAL "")
    set(FLATBUFFERS_HOME "/usr")
endif()

set(FLATBUFFERS_SEARCH_LIB_PATH
  ${FLATBUFFERS_HOME}/lib
  ${FLATBUFFERS_HOME}/lib/x86_64-linux-gnu
  ${FLATBUFFERS_HOME}/lib64
  ${FLATBUFFERS_HOME}/build
)

set(FLATBUFFERS_SEARCH_INCLUDE_DIR
  ${FLATBUFFERS_HOME}/include/flatbuffers/
)

find_path(FLATBUFFERS_INCLUDE_DIR flatbuffers.h
    PATHS ${FLATBUFFERS_SEARCH_INCLUDE_DIR}
    NO_DEFAULT_PATH
    DOC "Path to flatbuffers headers"
)

find_library(FLATBUFFERS_STATIC_LIB NAMES libflatbuffers.a
    PATHS ${FLATBUFFERS_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH
    DOC "Path to flatbuffers static library"
)

set(FLATBUFFERS_FLATC_EXECUTABLE ${FLATBUFFERS_HOME}/bin/flatc)
find_program(FLATBUFFERS_FLATC_EXECUTABLE NAMES flatc)

if (NOT FLATBUFFERS_STATIC_LIB)
    message(FATAL_ERROR "flatbuffers includes and libraries NOT found. "
      "Looked for headers in ${FLATBUFFERS_SEARCH_INCLUDE_DIR}, "
      "and for libs in ${FLATBUFFERS_SEARCH_LIB_PATH}")
    set(FLATBUFFERS_FOUND FALSE)
else()
    set(FLATBUFFERS_INCLUDEDIR ${FLATBUFFERS_HOME}/include/)
    set(FLATBUFFERS_LIBDIR ${FLATBUFFERS_HOME}/lib)
    set(FLATBUFFERS_FOUND TRUE)
    add_library(flatbuffers STATIC IMPORTED)
    set_target_properties(flatbuffers PROPERTIES IMPORTED_LOCATION "${FLATBUFFERS_STATIC_LIB}")
endif ()

mark_as_advanced(
    FLATBUFFERS_FOUND
    FLATBUFFERS_FLATC_EXECUTABLE
    FLATBUFFERS_INCLUDEDIR
    FLATBUFFERS_INCLUDE_DIR
    FLATBUFFERS_LIBS
    FLATBUFFERS_STATIC_LIB
    flatbuffers
)

set(FLATBUFFERS_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})
include(${FLATBUFFERS_CMAKE_DIR}/BuildFlatBuffers.cmake)
