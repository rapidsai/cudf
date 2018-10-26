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

set(ARROW_DOWNLOAD_BINARY_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/)

# Download and unpack arrow at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Arrow.CMakeLists.txt.cmake ${ARROW_DOWNLOAD_BINARY_DIR}/CMakeLists.txt COPYONLY)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${ARROW_DOWNLOAD_BINARY_DIR}
)

if(result)
    message(FATAL_ERROR "CMake step for arrow failed: ${result}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${ARROW_DOWNLOAD_BINARY_DIR}
)

if(result)
    message(FATAL_ERROR "Build step for arrow failed: ${result}")
endif()

# Locate the Arrow package.
# Requires that you build with:
#   -DARROW_ROOT:PATH=/path/to/arrow_install_dir
set(ARROW_ROOT ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow-install/usr/local/)

# Need ARROW_VERSION for setting correct ARROW_GENERATED_IPC_DIR
set(ARROW_VERSION "apache-arrow-0.10.0")
if (NOT "$ENV{PARQUET_ARROW_VERSION}" STREQUAL "")
    set(ARROW_VERSION "$ENV{PARQUET_ARROW_VERSION}")
endif()
message(STATUS "C: ARROW_VERSION=${ARROW_VERSION}")

# Copy the arrow-format flatbuffer headers to include/ipc using configure_file (will sync if input file changes)
if ("${ARROW_VERSION}" STREQUAL "apache-arrow-0.7.1")
  set(ARROW_GENERATED_IPC_DIR ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow/cpp/src/arrow/ipc/)
else()
  set(ARROW_GENERATED_IPC_DIR ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow-build/src/arrow/ipc/)
endif()

configure_file(${ARROW_GENERATED_IPC_DIR}/File_generated.h ${CMAKE_SOURCE_DIR}/include/gdf/ipc/File_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Message_generated.h ${CMAKE_SOURCE_DIR}/include/gdf/ipc/Message_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Schema_generated.h ${CMAKE_SOURCE_DIR}/include/gdf/ipc/Schema_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Tensor_generated.h ${CMAKE_SOURCE_DIR}/include/gdf/ipc/Tensor_generated.h COPYONLY)

# Add transitive dependency: Flatbuffers
set(FLATBUFFERS_ROOT ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow-build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/)

include_directories(${FLATBUFFERS_ROOT}/include/)
link_directories(${FLATBUFFERS_ROOT}/lib/)
