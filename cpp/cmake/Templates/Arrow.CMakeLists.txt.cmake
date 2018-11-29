#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

cmake_minimum_required(VERSION 3.12)

project(arrow-download NONE)

include(ExternalProject)

#NOTE
# libcudf.so` is now built with the old ABI `-D_GLIBCXX_USE_CXX11_ABI=0`
# If you build Arrow from source, you can fix this by using `-DARROW_TENSORFLOW=ON`.
# This forces Arrow to use the old ABI.

ExternalProject_Add(arrow
    GIT_REPOSITORY    https://github.com/apache/arrow.git
    GIT_TAG           apache-arrow-0.11.1
    SOURCE_DIR        "${ARROW_ROOT}/arrow"
    SOURCE_SUBDIR     "cpp"
    BINARY_DIR        "${ARROW_ROOT}/build"
    INSTALL_DIR       "${ARROW_ROOT}/install"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ${ARROW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ARROW_ROOT}/install
) 

