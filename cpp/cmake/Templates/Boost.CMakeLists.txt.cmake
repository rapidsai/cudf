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

project(boost-download NONE)

include(ExternalProject)

# NOTE build Boost with old C++ ABI _GLIBCXX_USE_CXX11_ABI=0 and with -fPIC
ExternalProject_Add(boost
    URL               http://archive.ubuntu.com/ubuntu/pool/main/b/boost1.58/boost1.58_1.58.0+dfsg.orig.tar.gz
    DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-download"
    PREFIX            "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-src"
    BUILD_IN_SOURCE   1
    INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-install"
    CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=system,filesystem,regex,atomic,chrono,container,context,thread --with-icu --prefix=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-install
    BUILD_COMMAND     ./b2 install variant=release define=_GLIBCXX_USE_CXX11_ABI=0 stage cxxflags=-fPIC cflags=-fPIC link=static runtime-link=static threading=multi --exec-prefix=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-install --prefix=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-install -a
    INSTALL_COMMAND   ""
    UPDATE_COMMAND    ""
)
