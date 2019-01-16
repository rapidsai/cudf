#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# - Find BlazingDB protocol C++ library libblazingdb-protocol (libblazingdb-protocol.a)
# BLAZINGDB_PROTOCOL_ROOT hints the location
#
# This module defines
# BLAZINGDB_PROTOCOL_FOUND
# BLAZINGDB_PROTOCOL_INCLUDEDIR Preferred include directory e.g. <prefix>/include
# BLAZINGDB_PROTOCOL_INCLUDE_DIR, directory containing blazingdb-protocol headers
# BLAZINGDB_PROTOCOL_LIBS, blazingdb-protocol libraries
# BLAZINGDB_PROTOCOL_LIBDIR, directory containing blazingdb-protocol libraries
# BLAZINGDB_PROTOCOL_STATIC_LIB, path to blazingdb-protocol.a
# blazingdb-protocol - static library

# If BLAZINGDB_PROTOCOL_ROOT is not defined try to search in the default system path
if ("${BLAZINGDB_PROTOCOL_ROOT}" STREQUAL "")
    set(BLAZINGDB_PROTOCOL_ROOT "/usr")
endif()

set(BLAZINGDB_PROTOCOL_SEARCH_LIB_PATH
  ${BLAZINGDB_PROTOCOL_ROOT}/lib
  ${BLAZINGDB_PROTOCOL_ROOT}/lib/x86_64-linux-gnu
  ${BLAZINGDB_PROTOCOL_ROOT}/lib64
  ${BLAZINGDB_PROTOCOL_ROOT}/build
)

set(BLAZINGDB_PROTOCOL_SEARCH_INCLUDE_DIR
  ${BLAZINGDB_PROTOCOL_ROOT}/include/blazingdb/protocol/
)

find_path(BLAZINGDB_PROTOCOL_INCLUDE_DIR api.h
    PATHS ${BLAZINGDB_PROTOCOL_SEARCH_INCLUDE_DIR}
    NO_DEFAULT_PATH
    DOC "Path to blazingdb-protocol headers"
)

#find_library(BLAZINGDB_PROTOCOL_LIBS NAMES blazingdb-protocol
#    PATHS ${BLAZINGDB_PROTOCOL_SEARCH_LIB_PATH}
#    NO_DEFAULT_PATH
#    DOC "Path to blazingdb-protocol library"
#)

find_library(BLAZINGDB_PROTOCOL_STATIC_LIB NAMES libblazingdb-protocol.a
    PATHS ${BLAZINGDB_PROTOCOL_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH
    DOC "Path to blazingdb-protocol static library"
)

if (NOT BLAZINGDB_PROTOCOL_STATIC_LIB)
    message(FATAL_ERROR "blazingdb-protocol includes and libraries NOT found. "
      "Looked for headers in ${BLAZINGDB_PROTOCOL_SEARCH_INCLUDE_DIR}, "
      "and for libs in ${BLAZINGDB_PROTOCOL_SEARCH_LIB_PATH}")
    set(BLAZINGDB_PROTOCOL_FOUND FALSE)
else()
    set(BLAZINGDB_PROTOCOL_INCLUDEDIR ${BLAZINGDB_PROTOCOL_ROOT}/include/)
    set(BLAZINGDB_PROTOCOL_LIBDIR ${BLAZINGDB_PROTOCOL_ROOT}/build) # TODO percy make this part cross platform
    set(BLAZINGDB_PROTOCOL_FOUND TRUE)
    add_library(blazingdb-protocol STATIC IMPORTED)
    set_target_properties(blazingdb-protocol PROPERTIES IMPORTED_LOCATION "${BLAZINGDB_PROTOCOL_STATIC_LIB}")
endif ()

mark_as_advanced(
  BLAZINGDB_PROTOCOL_FOUND
  BLAZINGDB_PROTOCOL_INCLUDEDIR
  BLAZINGDB_PROTOCOL_INCLUDE_DIR
  #BLAZINGDB_PROTOCOL_LIBS
  BLAZINGDB_PROTOCOL_STATIC_LIB
  blazingdb-protocol
)
