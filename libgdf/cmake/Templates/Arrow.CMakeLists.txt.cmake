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

project(arrow-download NONE)

include(ExternalProject)

set(ARROW_VERSION "apache-arrow-0.10.0")

if (NOT "$ENV{PARQUET_ARROW_VERSION}" STREQUAL "")
    set(ARROW_VERSION "$ENV{PARQUET_ARROW_VERSION}")
endif()

message(STATUS "Using Apache Arrow version: ${ARROW_VERSION}")

# Enable using arrow forks:
set(ARROW_GITHUB "https://github.com/apache/arrow")
if (NOT "$ENV{ARROW_GITHUB}" STREQUAL "")
    set(ARROW_GITHUB "$ENV{ARROW_GITHUB}")
endif()

set(ARROW_URL "${ARROW_GITHUB}/archive/${ARROW_VERSION}.tar.gz")

set(ARROW_CMAKE_ARGS
    #Arrow dependencies
    -DARROW_WITH_LZ4=OFF
    -DARROW_WITH_ZSTD=OFF
    -DARROW_WITH_BROTLI=OFF
    -DARROW_WITH_SNAPPY=OFF
    -DARROW_WITH_ZLIB=OFF

    #Build settings
    -DARROW_BUILD_STATIC=ON
    -DARROW_BUILD_SHARED=ON
    -DARROW_BOOST_USE_SHARED=ON
    -DARROW_BUILD_TESTS=OFF
    -DARROW_TEST_MEMCHECK=OFF
    -DARROW_BUILD_BENCHMARKS=OFF

    #Arrow modules
    -DARROW_IPC=ON
    -DARROW_COMPUTE=ON
    -DARROW_GPU=ON
    -DARROW_JEMALLOC=OFF
    -DARROW_BOOST_VENDORED=OFF
    -DARROW_PYTHON=ON
)

if (${ARROW_VERSION} STREQUAL "apache-arrow-0.9.0")
  # Keep ARROW_HDFS=ON to workaround arrow-0.9 bug that disables
  # boost_regex. See https://issues.apache.org/jira/browse/ARROW-2903
elseif (${ARROW_VERSION} STREQUAL "master")
  # pyarrow requires HDFS, so enabling it explicitly:
  set(ARROW_CMAKE_ARGS ${ARROW_CMAKE_ARGS} -DARROW_HDFS=ON)
else ()
  set(ARROW_CMAKE_ARGS ${ARROW_CMAKE_ARGS} -DARROW_HDFS=OFF)
endif()

ExternalProject_Add(arrow
    URL                ${ARROW_URL}
    CONFIGURE_COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" ${ARROW_CMAKE_ARGS} "${CMAKE_CURRENT_BINARY_DIR}/arrow-prefix/src/arrow/cpp/"
    INSTALL_COMMAND   make DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/arrow-prefix/src/arrow-install install
)
