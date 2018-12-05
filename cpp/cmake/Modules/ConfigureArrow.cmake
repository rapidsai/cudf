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

set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)

set(ARROW_CMAKE_ARGS " -DARROW_WITH_LZ4=OFF"
                     " -DARROW_WITH_ZSTD=OFF"
                     " -DARROW_WITH_BROTLI=OFF"
                     " -DARROW_WITH_SNAPPY=OFF"
                     " -DARROW_WITH_ZLIB=OFF"
                     " -DARROW_BUILD_STATIC=ON"
                     " -DARROW_BUILD_SHARED=OFF"
                     " -DARROW_BOOST_USE_SHARED=ON"
                     " -DARROW_BUILD_TESTS=OFF"
                     " -DARROW_TEST_MEMCHECK=OFF"
                     " -DARROW_BUILD_BENCHMARKS=OFF"
                     " -DARROW_IPC=ON"
                     " -DARROW_COMPUTE=OFF"
                     " -DARROW_GPU=OFF"
                     " -DARROW_JEMALLOC=OFF"
                     " -DARROW_BOOST_VENDORED=OFF"
                     " -DARROW_PYTHON=OFF"
                     " -DARROW_HDFS=ON"
                     " -DARROW_TENSORFLOW=ON" # enable old ABI for C/C++
                     ) 

# Download and unpack arrow at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Arrow.CMakeLists.txt.cmake ${ARROW_ROOT}/download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE ARROW_CONFIG
    WORKING_DIRECTORY ${ARROW_ROOT}/download/
)

if(ARROW_CONFIG)
    message(FATAL_ERROR "Configuring Arrow failed: " ${ARROW_CONFIG})
endif(ARROW_CONFIG)


# Parallel builds cause Travis to run out of memory
unset(PARALLEL_BUILD)            
if($ENV{TRAVIS})
    if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        message(STATUS "Disabling Parallel CMake build on Travis")
    else()
        set(PARALLEL_BUILD --parallel)
        message(STATUS "Using $ENV{CMAKE_BUILD_PARALLEL_LEVEL} build jobs on Travis")
    endif(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
else()
    set(PARALLEL_BUILD --parallel)
    message("STATUS Enabling Parallel CMake build")
endif($ENV{TRAVIS})
 
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${PARALLEL_BUILD} .
    RESULT_VARIABLE ARROW_BUILD
    WORKING_DIRECTORY ${ARROW_ROOT}/download)

if(ARROW_BUILD)
    message(FATAL_ERROR "Build step for Arrow failed: ${ARROW_BUILD}")
endif(ARROW_BUILD)


# Add transitive dependency: Flatbuffers
set(FLATBUFFERS_ROOT ${ARROW_ROOT}/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/)
message(STATUS "FlatBuffers installed here: " ${FLATBUFFERS_ROOT})
set(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_ROOT}/include")
set(FLATBUFFERS_LIBRARY_DIR "${FLATBUFFERS_ROOT}/lib")

add_definitions(-DARROW_METADATA_V4)
add_definitions(-DARROW_VERSION=1101) # DARROW_VERSION=1101 is related to apache-arrow-0.11.1

include_directories(${FLATBUFFERS_INCLUDE_DIR})
link_directories(${FLATBUFFERS_LIBRARY_DIR})

set(ARROW_GENERATED_IPC_DIR ${ARROW_ROOT}/build/src/arrow/ipc/)
configure_file(${ARROW_GENERATED_IPC_DIR}/File_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/File_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Message_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Message_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Schema_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Schema_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Tensor_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Tensor_generated.h COPYONLY)

set(ENV{ARROW_HOME} ${ARROW_ROOT}/install/)