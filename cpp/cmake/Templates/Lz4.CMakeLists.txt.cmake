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

cmake_minimum_required(VERSION 3.12)

project(lz4-download NONE)

include(ExternalProject)

ExternalProject_Add(lz4
    GIT_REPOSITORY    https://github.com/lz4/lz4.git
    GIT_TAG           v1.7.5
    SOURCE_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-src"
    BUILD_IN_SOURCE   1
    INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-install"
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND    ""
    BUILD_COMMAND     ${CMAKE_MAKE_PROGRAM} -j4 install
)
