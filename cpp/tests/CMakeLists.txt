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
# ##################################################################################################

# ##################################################################################################
# * io tests --------------------------------------------------------------------------------------
ConfigureTest(DECOMPRESSION_TEST io/comp/decomp_test.cpp)
ConfigureTest(ROW_SELECTION_TEST io/row_selection_test.cpp)

ConfigureTest(
  CSV_TEST io/csv_test.cpp
  GPUS 1
  PERCENT 30 EXTRA_LIBS ${ARROW_LIBRARIES}
)
ConfigureTest(
  FILE_IO_TEST io/file_io_test.cpp
  GPUS 1
  PERCENT 30
)
ConfigureTest(
  ORC_TEST io/orc_chunked_reader_test.cu io/orc_test.cpp
  GPUS 1
  PERCENT 30
)
ConfigureTest(
  PARQUET_TEST
  io/parquet_test.cpp
  io/parquet_chunked_reader_test.cu
  io/parquet_chunked_writer_test.cpp
  io/parquet_common.cpp
  io/parquet_misc_test.cpp
  io/parquet_reader_test.cpp
  io/parquet_writer_test.cpp
  io/parquet_v2_test.cpp
  GPUS 1
  PERCENT 30
)
ConfigureTest(
  JSON_TEST io/json/json_test.cpp io/json/json_chunked_reader.cu
  GPUS 1
  PERCENT 30 EXTRA_LIBS ${ARROW_LIBRARIES}
)
ConfigureTest(JSON_WRITER_TEST io/json/json_writer.cpp)
ConfigureTest(JSON_TYPE_CAST_TEST io/json/json_type_cast_test.cu)
ConfigureTest(NESTED_JSON_TEST io/json/nested_json_test.cpp io/json/json_tree.cpp)
ConfigureTest(MULTIBYTE_SPLIT_TEST io/text/multibyte_split_test.cpp)
ConfigureTest(JSON_QUOTE_NORMALIZATION io/json/json_quote_normalization_test.cpp)
ConfigureTest(JSON_WHITESPACE_NORMALIZATION io/json/json_whitespace_normalization_test.cu)
ConfigureTest(JSON_TREE_CSR io/json/json_tree_csr.cu)
ConfigureTest(
  DATA_CHUNK_SOURCE_TEST io/text/data_chunk_source_test.cpp
  GPUS 1
  PERCENT 30
)
target_link_libraries(DATA_CHUNK_SOURCE_TEST PRIVATE ZLIB::ZLIB)
ConfigureTest(LOGICAL_STACK_TEST io/fst/logical_stack_test.cu)
ConfigureTest(FST_TEST io/fst/fst_test.cu)
ConfigureTest(TYPE_INFERENCE_TEST io/type_inference_test.cu)

# ##################################################################################################
# * utilities tests -------------------------------------------------------------------------------
ConfigureTest(
  UTILITIES_TEST
  utilities_tests/batched_memcpy_tests.cu
  utilities_tests/batched_memset_tests.cu
  utilities_tests/column_debug_tests.cpp
  utilities_tests/column_utilities_tests.cpp
  utilities_tests/column_wrapper_tests.cpp
  utilities_tests/default_stream_tests.cpp
  utilities_tests/io_utilities_tests.cpp
  utilities_tests/lists_column_wrapper_tests.cpp
  utilities_tests/logger_tests.cpp
  utilities_tests/pinned_memory_tests.cpp
  utilities_tests/type_check_tests.cpp
  utilities_tests/type_list_tests.cpp
)

# ##################################################################################################
# * span tests -------------------------------------------------------------------------------

# This test must be split into two executables so that one can use the preload library and one does
# not. The one that doesn't includes a thrust::device_vector copy, which is always synchronous on
# the default stream and is out of libcudf's control (but must be tested).
set(_allowlist_filter SpanTest.CanConstructFromDeviceContainers)

ConfigureTest(SPAN_TEST utilities_tests/span_tests.cu)
ConfigureTest(SPAN_TEST_DEVICE_VECTOR utilities_tests/span_tests.cu)

# Overwrite the environments set by ConfigureTest
set_property(
  TEST SPAN_TEST SPAN_TEST_DEVICE_VECTOR
  APPEND
  PROPERTY ENVIRONMENT "GTEST_FILTER=-${_allowlist_filter}"
)

# ##################################################################################################
# * stream testing ---------------------------------------------------------------------------------
if(CUDF_BUILD_STREAMS_TEST_UTIL)
  ConfigureTest(
    STREAM_IDENTIFICATION_TEST identify_stream_usage/test_default_stream_identification.cu
  )
endif()

ConfigureTest(STREAM_BINARYOP_TEST streams/binaryop_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_CONCATENATE_TEST streams/concatenate_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_COPYING_TEST streams/copying_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_CSVIO_TEST streams/io/csv_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_DATETIME_TEST streams/datetime_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_DICTIONARY_TEST streams/dictionary_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_FILLING_TEST streams/filling_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_GROUPBY_TEST streams/groupby_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_HASHING_TEST streams/hash_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_JOIN_TEST streams/join_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_JSONIO_TEST streams/io/json_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_LABELING_BINS_TEST streams/labeling_bins_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_LISTS_TEST streams/lists_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_MERGE_TEST streams/merge_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_MULTIBYTE_SPLIT_TEST streams/io/multibyte_split_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_NULL_MASK_TEST streams/null_mask_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_ORCIO_TEST streams/io/orc_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_PARQUETIO_TEST streams/io/parquet_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_PARTITIONING_TEST streams/partitioning_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_POOL_TEST streams/pool_test.cu STREAM_MODE testing)
ConfigureTest(STREAM_QUANTILE_TEST streams/quantile_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_REDUCTION_TEST streams/reduction_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_REPLACE_TEST streams/replace_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_RESHAPE_TEST streams/reshape_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_ROLLING_TEST streams/rolling_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_ROUND_TEST streams/round_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_SEARCH_TEST streams/search_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_SORTING_TEST streams/sorting_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_STREAM_COMPACTION_TEST streams/stream_compaction_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_STRINGS_TEST
  streams/strings/case_test.cpp
  streams/strings/combine_test.cpp
  streams/strings/contains_test.cpp
  streams/strings/convert_test.cpp
  streams/strings/extract_test.cpp
  streams/strings/factory_test.cpp
  streams/strings/filter_test.cpp
  streams/strings/find_test.cpp
  streams/strings/replace_test.cpp
  streams/strings/reverse_test.cpp
  streams/strings/split_test.cpp
  streams/strings/strings_tests.cpp
  STREAM_MODE
  testing
)
ConfigureTest(
  STREAM_TEXT_TEST
  streams/text/edit_distance_test.cpp
  streams/text/ngrams_test.cpp
  streams/text/replace_test.cpp
  streams/text/stemmer_test.cpp
  streams/text/subword_tokenize_test.cpp
  streams/text/tokenize_test.cpp
  STREAM_MODE
  testing
)
ConfigureTest(STREAM_TRANSFORM_TEST streams/transform_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_TRANSPOSE_TEST streams/transpose_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_UNARY_TEST streams/unary_test.cpp STREAM_MODE testing)

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf)
