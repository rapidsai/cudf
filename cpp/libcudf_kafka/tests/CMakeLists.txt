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
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME )
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    set_target_properties(${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDA_KAFKA_BINARY_DIR}/gtests>")
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE GTest::gmock_main GTest::gtest_main cudf_kafka)
    target_include_directories(${CMAKE_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
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
