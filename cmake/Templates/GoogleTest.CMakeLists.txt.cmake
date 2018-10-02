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
cmake_minimum_required(VERSION 2.8.12)

cmake_policy(SET CMP0048 NEW)

project(googletest-download NONE)

include(ExternalProject)

ExternalProject_Add(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           release-1.8.1
    SOURCE_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-src"
    BINARY_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-build"
    INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-install"
    CMAKE_ARGS        -Dgtest_build_samples=ON -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/googletest-install
)
