# =============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
function(ConfigureTest CMAKE_TEST_NAME)
  set(options)
  set(one_value GPUS PERCENT STREAM_MODE)
  set(multi_value EXTRA_LIBS)
  cmake_parse_arguments(_CUDF_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _CUDF_TEST_GPUS AND NOT DEFINED _CUDF_TEST_PERCENT)
    set(_CUDF_TEST_GPUS 1)
    set(_CUDF_TEST_PERCENT 15)
  endif()
  if(NOT DEFINED _CUDF_TEST_GPUS)
    set(_CUDF_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUDF_TEST_PERCENT)
    set(_CUDF_TEST_PERCENT 100)
  endif()
  if(NOT DEFINED _CUDF_TEST_STREAM_MODE)
    set(_CUDF_TEST_STREAM_MODE cudf)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${_CUDF_TEST_UNPARSED_ARGUMENTS})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )

  target_link_libraries(
    ${CMAKE_TEST_NAME}
    PRIVATE cudf::cudftestutil
            cudf::cudftestutil_impl
            GTest::gmock
            GTest::gmock_main
            GTest::gtest
            GTest::gtest_main
            nvtx3::nvtx3-cpp
            $<TARGET_NAME_IF_EXISTS:conda_env>
            "${_CUDF_TEST_EXTRA_LIBS}"
  )
  rapids_cuda_set_runtime(${CMAKE_TEST_NAME} USE_STATIC ${CUDA_STATIC_RUNTIME})
  rapids_test_add(
    NAME ${CMAKE_TEST_NAME}
    COMMAND ${CMAKE_TEST_NAME}
    GPUS ${_CUDF_TEST_GPUS}
    PERCENT ${_CUDF_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  if(CUDF_BUILD_STREAMS_TEST_UTIL)
    set_tests_properties(
      ${CMAKE_TEST_NAME}
      PROPERTIES
        ENVIRONMENT
        "GTEST_CUDF_STREAM_MODE=new_${_CUDF_TEST_STREAM_MODE}_default;LD_PRELOAD=$<TARGET_FILE:cudf_identify_stream_usage_mode_${_CUDF_TEST_STREAM_MODE}>"
    )
  endif()
endfunction()

# ##################################################################################################
# dependencies  ###################################################################################
# ##################################################################################################

# No need to install Arrow libs when only the final test executables are shipped.
set(CUDF_EXCLUDE_ARROW_FROM_ALL ON)
include(../cmake/thirdparty/get_arrow.cmake)

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################ConfigureTest(STREAM_CSVIO_TEST streams/io/csv_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_CSVIO_TEST streams/io/csv_test.cpp STREAM_MODE testing)

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf)
