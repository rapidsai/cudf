#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

# find_package(Threads REQUIRED)

# add_library(cudftestutil STATIC
#             utilities/base_fixture.cpp
#             utilities/column_utilities.cu
#             utilities/table_utilities.cu
#             strings/utilities.cu)

# target_compile_options(cudftestutil
#             PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
#                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
# )

# target_compile_features(cudftestutil PUBLIC cxx_std_14 cuda_std_14)

# target_link_libraries(cudftestutil
#                PUBLIC GTest::gmock
#                       GTest::gtest
#                       Threads::Threads
#                       cudf)

# target_include_directories(cudftestutil
#     PUBLIC "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
#            "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>")

# install(TARGETS cudftestutil
#         DESTINATION lib
#         EXPORT cudf-targets)

include(../cmake/thirdparty/CUDF_KAFKA_GetCUDF.cmake)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME )
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    set_target_properties(${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gtests>")
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE cudf::cudftestutil GTest::gmock_main GTest::gtest_main)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction()

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - Kafka host tests ----------------------------------------------------------------------------------
# ConfigureTest(KAFKA_HOST_TEST
#     kafka_consumer_tests.cpp)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()







































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

project(KAFKA_TESTS VERSION 0.19.0 LANGUAGES C CXX CUDA)


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
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(CUDF_KAFKA_TEST_LIST CACHE INTERNAL "CUDF_KAFKA_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                    ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gtest_main pthread cuda cudf_kafka)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(CUDF_KAFKA_TEST_LIST ${CUDF_KAFKA_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CUDF_KAFKA_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

###################################################################################################
# - create tests ----------------------------------------------------------------------------------
ConfigureTest(CUDF_KAFKA_HOST_READ  kafka_consumer_tests.cpp)

enable_testing()
