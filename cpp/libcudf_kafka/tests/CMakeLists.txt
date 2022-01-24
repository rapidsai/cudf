# =============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest test_name)
  add_executable(${test_name} ${ARGN})
  set_target_properties(
    ${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                            "$<BUILD_INTERFACE:${CUDA_KAFKA_BINARY_DIR}/gtests>"
  )
  target_link_libraries(
    ${test_name} PRIVATE GTest::gmock GTest::gmock_main GTest::gtest_main cudf_kafka
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# ##################################################################################################
# * Kafka host tests
# ----------------------------------------------------------------------------------
ConfigureTest(KAFKA_HOST_TEST kafka_consumer_tests.cpp)
