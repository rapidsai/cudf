# =============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
# enable testing ################################################################################
# ##################################################################################################
enable_testing()

include(rapids-test)
rapids_test_init()

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest test_name)
  add_executable(${test_name} ${ARGN})
  set_target_properties(
    ${test_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_KAFKA_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )
  target_link_libraries(
    ${test_name} PRIVATE GTest::gmock GTest::gmock_main GTest::gtest_main cudf_kafka
  )
  rapids_test_add(
    NAME ${test_name}
    COMMAND ${test_name}
    GPUS 1
    PERCENT 25
    INSTALL_COMPONENT_SET testing
  )
endfunction()

# ##################################################################################################
# * Kafka host tests
# ----------------------------------------------------------------------------------
ConfigureTest(KAFKA_HOST_TEST kafka_consumer_tests.cpp)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf_kafka)
