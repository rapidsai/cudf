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

cmake_minimum_required(VERSION 3.11)

project(arrow-download NONE)

include(ExternalProject)

message(STATUS "Using Apache Arrow version: ${ARROW_VERSION}")

#NOTE
# libcudf.so` is now built with the old ABI `-D_GLIBCXX_USE_CXX11_ABI=0`
# If you build Arrow from source, you can fix this by using `-DARROW_TENSORFLOW=ON`.
# This forces Arrow to use the old ABI.

ExternalProject_Add(arrow
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-install
        -DARROW_IPC=ON
        -DARROW_HDFS=ON
        -DARROW_TENSORFLOW=ON
    GIT_REPOSITORY    https://github.com/apache/arrow.git
    GIT_TAG           ${ARROW_VERSION}
    UPDATE_COMMAND    ""
    SOURCE_SUBDIR     cpp
    BINARY_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-build"
    INSTALL_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-install"
    SOURCE_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-src"
) 
