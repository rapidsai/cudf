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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_DATAFRAME_EXTERNAL VERSION 0.14.0 LANGUAGES C CXX)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "EXTERNAL DATASOURCE: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "EXTERNAL DATASOURCE: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

# Build options
option(BUILD_TESTS "Configure CMake to build tests" ON)

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/kafka/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)

    message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")
    message(STATUS "NVSTRINGS_TEST_LIST set to: ${NVSTRINGS_TEST_LIST}")
endif(BUILD_TESTS)

###################################################################################################
# - set libcudf headers ---------------------------------------------------------------------------

set(CUDF_INCLUDE "${PROJECT_SOURCE_DIR}/../cpp/src/io/utilities")
message(STATUS "CUDF_INCLUDE: CUDF_INCLUDE set to ${CUDF_INCLUDE}")

###################################################################################################
# - find librdkafka -------------------------------------------------------------------------------

find_path(RDKAFKA_INCLUDE "librdkafka"
        HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++"
        HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")

message(STATUS "RDKAFKA: RDKAFKA++_LIBRARY set to ${RDKAFKA++_LIBRARY}")
message(STATUS "RDKAFKA: RDKAFKA_INCLUDE set to ${RDKAFKA_INCLUDE}")

#add_library(rdkafka SHARED IMPORTED ${RDKAFKA_LIBRARY})
add_library(rdkafka++ SHARED IMPORTED ${RDKAFKA++_LIBRARY})
if(RDKAFKA_INCLUDE AND RDKAFKA++_LIBRARY)
    set_target_properties(rdkafka++ PROPERTIES IMPORTED_LOCATION ${RDKAFKA++_LIBRARY})
endif(RDKAFKA_INCLUDE AND RDKAFKA++_LIBRARY)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/kafka/include"
                    "${CUDF_INCLUDE}")

if(CONDA_INCLUDE_DIRS)
    include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf_kafka SHARED
            kafka/src/kafka_consumer.cpp
            # kafka/src/kafka_producer.cpp
)

# Include librdkafka exclusively for libcudf_kafka
target_include_directories(cudf_kafka PUBLIC "${RDKAFKA_INCLUDE}")
target_link_directories(cudf_kafka PUBLIC "${RDKAFKA++_LIBRARY}")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

# link targets for cudf_kafka
target_link_libraries(cudf_kafka rdkafka++)

install(TARGETS cudf_kafka
        DESTINATION lib/external
        COMPONENT cudf_kafka)

install(TARGETS cudf_kafka
        DESTINATION lib
        COMPONENT cudf_kafka)

add_custom_target(build_tests_cudf
                  DEPENDS ${CUDF_TEST_LIST})
