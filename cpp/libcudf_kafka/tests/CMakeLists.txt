# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
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
