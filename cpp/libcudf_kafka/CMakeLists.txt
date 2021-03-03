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
project(CUDA_KAFKA VERSION 0.19.0 LANGUAGES C CXX CUDA)

#################################################################################################
# - CUDA architectures

# If CMAKE_CUDA_ARCHITECTURES is not defined, build for all supported architectures. If
# CMAKE_CUDA_ARCHITECTURES is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of CMAKE_CUDA_ARCHITECTURES, https://gitlab.kitware.com/cmake/cmake/-/issues/21302
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CUDF_BUILD_FOR_ALL_ARCHS TRUE)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
  set(CUDF_BUILD_FOR_DETECTED_ARCHS TRUE)
endif()

###################################################################################################
# - Build options

option(CUDF_KAFKA_BUILD_TESTS "Build tests for libcudf_kafka" ON)
option(ARROW_STATIC_LIB "Build static arrow library" OFF)

message(VERBOSE "CUDF_KAFKA: Build gtests: ${CUDF_KAFKA_BUILD_TESTS}")
message(VERBOSE "CUDF_KAFKA: Build Arrow static library: ${ARROW_STATIC_LIB}")

###################################################################################################
# - Dependencies

# CPM
include(cmake/thirdparty/CUDF_KAFKA_GetCPM.cmake)

# libcudf
include(cmake/thirdparty/CUDF_KAFKA_GetCUDF.cmake)

# # librdkafka
# include(cmake/thirdparty/CUDF_KAFKA_GetRDKafka.cmake)


###################################################################################################
# - tests and benchmarks --------------------------------------------------------------------------

# if (CUDF_KAFKA_BUILD_TESTS)
#     # Find or install GoogleTest
#     CPMFindPackage(NAME GTest
#         VERSION         1.10.0
#         GIT_REPOSITORY  https://github.com/google/googletest.git
#         GIT_TAG         release-1.10.0
#         GIT_SHALLOW     TRUE
#         OPTIONS         "INSTALL_GTEST OFF"
#         # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
#         FIND_PACKAGE_ARGUMENTS "CONFIG")
#     # Add GTest aliases if they don't already exist.
#     # Assumes if GTest::gtest doesn't exist, the others don't either.
#     # TODO: Is this always a valid assumption?
#     if(NOT TARGET GTest::gtest)
#         add_library(GTest::gtest ALIAS gtest)
#         add_library(GTest::gmock ALIAS gmock)
#         add_library(GTest::gtest_main ALIAS gtest_main)
#         add_library(GTest::gmock_main ALIAS gmock_main)
#     endif()

#     # include CTest module -- automatically calls enable_testing()
#     include(CTest)
#     add_subdirectory(tests)
# endif()

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

# set_target_properties(cudf_kafka PROPERTIES BUILD_RPATH "\$ORIGIN")

# # Include paths
# include_directories("${CMAKE_SOURCE_DIR}/include"
#                     "${CMAKE_CURRENT_SOURCE_DIR}/include/cudf")

###################################################################################################
# cudf_kafka - librdkafka -------------------------------------------------------------------------

find_path(RDKAFKA_INCLUDE "librdkafka" HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++" HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")

target_link_libraries(cudf_kafka ${RDKAFKA++_LIBRARY})
include_directories("${RDKAFKA_INCLUDE}")

###################################################################################################
# - cudf_kafka Install ----------------------------------------------------------------------------
target_link_libraries(cudf_kafka cudf::cudf)

install(TARGETS cudf_kafka
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION include)

add_custom_target(build_tests_libcudf_kafka
                    DEPENDS ${CUDF_KAFKA_TEST_LIST})

add_custom_target(test_libcudf_kafka
                    COMMAND ctest
                    DEPENDS build_tests_libcudf_kafka)
