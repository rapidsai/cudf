# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#[=======================================================================[.rst:
FindcuFile
----------

Find cuFile headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``cufile::cuFile``
  The cuFile library, if found.
``cufile::cuFileRDMA``
  The cuFile RDMA library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``cuFile_FOUND``
  true if (the requested version of) cuFile is available.
``cuFile_VERSION``
  the version of cuFile.
``cuFile_LIBRARIES``
  the libraries to link against to use cuFile.
``cuFileRDMA_LIBRARIES``
  the libraries to link against to use cuFile RDMA.
``cuFile_INCLUDE_DIRS``
  where to find the cuFile headers.
``cuFile_COMPILE_OPTIONS``
  this should be passed to target_compile_options(), if the
  target is not used for linking

#]=======================================================================]

# use pkg-config to get the directories and then use these values in the FIND_PATH() and
# FIND_LIBRARY() calls
find_package(PkgConfig QUIET)
pkg_check_modules(PKG_cuFile QUIET cuFile)

set(cuFile_COMPILE_OPTIONS ${PKG_cuFile_CFLAGS_OTHER})
set(cuFile_VERSION ${PKG_cuFile_VERSION})

# Find the location of the CUDA Toolkit
find_package(CUDAToolkit QUIET)
find_path(
  cuFile_INCLUDE_DIR
  NAMES cufile.h
  HINTS ${PKG_cuFile_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
)

find_library(
  cuFile_LIBRARY
  NAMES cufile
  HINTS ${PKG_cuFile_LIBRARY_DIRS} ${CUDAToolkit_LIBRARY_DIR}
)

find_library(
  cuFileRDMA_LIBRARY
  NAMES cufile_rdma
  HINTS ${PKG_cuFile_LIBRARY_DIRS} ${CUDAToolkit_LIBRARY_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  cuFile
  FOUND_VAR cuFile_FOUND
  REQUIRED_VARS cuFile_LIBRARY cuFileRDMA_LIBRARY cuFile_INCLUDE_DIR
  VERSION_VAR cuFile_VERSION
)

if(cuFile_INCLUDE_DIR AND NOT TARGET cufile::cuFile_interface)
  add_library(cufile::cuFile_interface INTERFACE IMPORTED GLOBAL)
  target_include_directories(
    cufile::cuFile_interface INTERFACE "$<BUILD_INTERFACE:${cuFile_INCLUDE_DIR}>"
  )
  target_compile_options(cufile::cuFile_interface INTERFACE "${cuFile_COMPILE_OPTIONS}")
  target_compile_definitions(cufile::cuFile_interface INTERFACE CUFILE_FOUND)
endif()

if(cuFile_FOUND AND NOT TARGET cufile::cuFile)
  add_library(cufile::cuFile UNKNOWN IMPORTED GLOBAL)
  set_target_properties(
    cufile::cuFile
    PROPERTIES IMPORTED_LOCATION "${cuFile_LIBRARY}"
               INTERFACE_COMPILE_OPTIONS "${cuFile_COMPILE_OPTIONS}"
               INTERFACE_INCLUDE_DIRECTORIES "${cuFile_INCLUDE_DIR}"
  )
endif()

if(cuFile_FOUND AND NOT TARGET cufile::cuFileRDMA)
  add_library(cufile::cuFileRDMA UNKNOWN IMPORTED GLOBAL)
  set_target_properties(
    cufile::cuFileRDMA
    PROPERTIES IMPORTED_LOCATION "${cuFileRDMA_LIBRARY}"
               INTERFACE_COMPILE_OPTIONS "${cuFile_COMPILE_OPTIONS}"
               INTERFACE_INCLUDE_DIRECTORIES "${cuFile_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(cuFile_LIBRARY cuFileRDMA_LIBRARY cuFile_INCLUDE_DIR)

if(cuFile_FOUND)
  set(cuFile_LIBRARIES ${cuFile_LIBRARY})
  set(cuFileRDMA_LIBRARIES ${cuFileRDMA_LIBRARY})
  set(cuFile_INCLUDE_DIRS ${cuFile_INCLUDE_DIR})
endif()
