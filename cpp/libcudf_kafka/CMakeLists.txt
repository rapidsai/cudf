#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CUDA_KAFKA VERSION 0.19.0 LANGUAGES CXX)

###################################################################################################
# - Build options
option(BUILD_TESTS "Build tests for libcudf_kafka" ON)

message(VERBOSE "CUDF_KAFKA: Build gtests: ${BUILD_TESTS}")

###################################################################################################
# - Dependencies

# CPM
include(../cmake/thirdparty/CUDF_GetCPM.cmake)

# libcudf
include(cmake/thirdparty/CUDF_KAFKA_GetCUDF.cmake)

# librdkafka
include(cmake/thirdparty/CUDF_KAFKA_GetRDKafka.cmake)

# # GTests if enabled
if (BUILD_TESTS)
    # GoogleTest
    include(../cmake/thirdparty/CUDF_GetGTest.cmake)

    # include CTest module -- automatically calls enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

###################################################################################################
# - library target --------------------------------------------------------------------------------

add_library(cudf_kafka SHARED
    src/kafka_consumer.cpp
)

###################################################################################################
# - cudf_kafka Install ----------------------------------------------------------------------------
target_link_libraries(cudf_kafka cudf::cudf RDKAFKA::RDKAFKA)

install(TARGETS cudf_kafka
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION include)
