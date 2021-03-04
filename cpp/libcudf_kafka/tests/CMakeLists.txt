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

if(cudf_ADDED)
    message(STATUS "HECK YEAH WE FOUND IT!!!!")
else()
    message(STATUS "DAMN, DIDN'T FIND IT.....")
endif()

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
# - Kafka host tests ----------------------------------------------------------------------------------
ConfigureTest(KAFKA_HOST_TEST
    kafka_consumer_tests.cpp)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
