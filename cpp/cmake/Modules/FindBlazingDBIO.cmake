#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# - Find BlazingDB protocol C++ library libblazingdb-io (libblazingdb-io.a)
# BLAZINGDB_IO_ROOT hints the location
#
# This module defines
# BLAZINGDB_IO_FOUND
# BLAZINGDB_IO_INCLUDEDIR Preferred include directory e.g. <prefix>/include
# BLAZINGDB_IO_INCLUDE_DIR, directory containing blazingdb-io headers
# BLAZINGDB_IO_LIBS, blazingdb-io libraries
# BLAZINGDB_IO_LIBDIR, directory containing blazingdb-io libraries
# BLAZINGDB_IO_STATIC_LIB, path to blazingdb-io.a
# blazingdb-io - static library

# If BLAZINGDB_IO_ROOT is not defined try to search in the default system path
if ("${BLAZINGDB_IO_ROOT}" STREQUAL "")
    set(BLAZINGDB_IO_ROOT "/usr")
endif()

set(BLAZINGDB_IO_SEARCH_LIB_PATH
  ${BLAZINGDB_IO_ROOT}/lib
  ${BLAZINGDB_IO_ROOT}/lib/x86_64-linux-gnu
  ${BLAZINGDB_IO_ROOT}/lib64
  ${BLAZINGDB_IO_ROOT}/build
)

set(BLAZINGDB_IO_SEARCH_INCLUDE_DIR
  ${BLAZINGDB_IO_ROOT}/include/blazingdb/io/
)

find_path(BLAZINGDB_IO_INCLUDE_DIR FileSystemInterface.h
    PATHS ${BLAZINGDB_IO_SEARCH_INCLUDE_DIR}/FileSystem/
    NO_DEFAULT_PATH
    DOC "Path to blazingdb-io headers"
)

#find_library(BLAZINGDB_IO_LIBS NAMES blazingdb-io
#    PATHS ${BLAZINGDB_IO_SEARCH_LIB_PATH}
#    NO_DEFAULT_PATH
#    DOC "Path to blazingdb-io library"
#)

find_library(BLAZINGDB_IO_STATIC_LIB NAMES libblazingdb-io.a
    PATHS ${BLAZINGDB_IO_SEARCH_LIB_PATH}
    NO_DEFAULT_PATH
    DOC "Path to blazingdb-io static library"
)

if (NOT BLAZINGDB_IO_STATIC_LIB)
    message(FATAL_ERROR "blazingdb-io includes and libraries NOT found. "
      "Looked for headers in ${BLAZINGDB_IO_SEARCH_INCLUDE_DIR}, "
      "and for libs in ${BLAZINGDB_IO_SEARCH_LIB_PATH}")
    set(BLAZINGDB_IO_FOUND FALSE)
else()
    set(BLAZINGDB_IO_INCLUDE_DIR ${BLAZINGDB_IO_ROOT}/include/blazingdb/io/)
    set(BLAZINGDB_IO_INCLUDEDIR ${BLAZINGDB_IO_ROOT}/include/)
    set(BLAZINGDB_IO_LIBDIR ${BLAZINGDB_IO_ROOT}/build) # TODO percy make this part cross platform
    set(BLAZINGDB_IO_FOUND TRUE)
    #add_library(blazingdb-io STATIC IMPORTED)
    #set_target_properties(blazingdb-io PROPERTIES IMPORTED_LOCATION "${BLAZINGDB_IO_STATIC_LIB}")
endif ()

mark_as_advanced(
  BLAZINGDB_IO_FOUND
  BLAZINGDB_IO_INCLUDEDIR
  BLAZINGDB_IO_INCLUDE_DIR
  #BLAZINGDB_IO_LIBS
  BLAZINGDB_IO_STATIC_LIB
  #blazingdb-io
)
