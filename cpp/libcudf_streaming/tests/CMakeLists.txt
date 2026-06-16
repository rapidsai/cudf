# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

include(rapids-test)
rapids_test_init()

file(WRITE "${CUDF_STREAMING_BINARY_DIR}/CTestTestfile.cmake" "subdirs(\"tests\")\n")

set(_cudf_streaming_gtests_link "${CMAKE_CURRENT_BINARY_DIR}/gtests")
if(NOT EXISTS "${_cudf_streaming_gtests_link}")
  file(CREATE_LINK "${CUDF_STREAMING_BINARY_DIR}/gtests" "${_cudf_streaming_gtests_link}" SYMBOLIC)
endif()

add_library(cudf_streaming_test_sources OBJECT)
set_target_properties(
  cudf_streaming_test_sources
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_include_directories(
  cudf_streaming_test_sources PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                      "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(
  cudf_streaming_test_sources
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS};--expt-extended-lambda>
)
target_link_libraries(
  cudf_streaming_test_sources
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf::cudftestutil cudf::cudftestutil_impl
  PUBLIC GTest::gmock GTest::gtest
)
target_sources(
  cudf_streaming_test_sources
  PRIVATE streaming/test_table_chunk.cpp
          streaming/test_read_parquet.cpp
          streaming/test_channel_metadata.cpp
          streaming/test_partition.cpp
          streaming/test_cudf_utils.cpp
          streaming/test_shuffler.cpp
          streaming/test_leaf_actor.cpp
          test_partition.cpp
          test_shuffler.cpp
          test_shuffler_many_streams.cpp
)

add_executable(cudf_streaming_single_tests main/single.cpp)
set_target_properties(
  cudf_streaming_single_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
             INSTALL_RPATH "\$ORIGIN/../../../lib"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_include_directories(
  cudf_streaming_single_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                      "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  cudf_streaming_single_tests
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf GTest::gmock GTest::gtest
          $<TARGET_NAME_IF_EXISTS:conda_env> cudf_streaming_test_sources
)
rapids_test_add(
  NAME cudf_streaming_single_tests
  COMMAND cudf_streaming_single_tests
  GPUS 1
  PERCENT 100
  INSTALL_COMPONENT_SET testing
)

rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf_streaming
)
