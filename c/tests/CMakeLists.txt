# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# C API test suite — pure C tests to verify header correctness and basic API

add_executable(test_headers core/headers.c)
target_include_directories(
  test_headers PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_compile_options(test_headers PRIVATE -std=c11)
add_test(NAME headers_compile COMMAND test_headers)

add_executable(test_c_api core/c_api.c)
target_include_directories(
  test_c_api PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_link_libraries(test_c_api cudf_c)
target_compile_options(test_c_api PRIVATE -std=c11)
add_test(NAME c_api_basic COMMAND test_c_api)

add_executable(test_arrow interop/arrow_test.c)
target_include_directories(
  test_arrow PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_link_libraries(test_arrow PRIVATE cudf_c)
target_compile_options(test_arrow PRIVATE -std=c11)
add_test(NAME arrow_interop COMMAND test_arrow)
set_tests_properties(arrow_interop PROPERTIES LABELS "gpu")

add_executable(test_parquet io/parquet_test.c)
target_include_directories(
  test_parquet PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_link_libraries(test_parquet PRIVATE cudf_c)
target_compile_options(test_parquet PRIVATE -std=c11)
add_test(NAME parquet_io COMMAND test_parquet)
set_tests_properties(parquet_io PROPERTIES LABELS "gpu")

add_executable(test_binaryop compute/binaryop_test.c)
target_include_directories(
  test_binaryop PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_link_libraries(test_binaryop PRIVATE cudf_c)
target_compile_options(test_binaryop PRIVATE -std=c11)
add_test(NAME binaryop_compute COMMAND test_binaryop)
set_tests_properties(binaryop_compute PROPERTIES LABELS "gpu")

add_executable(test_join compute/join_test.c)
target_include_directories(
  test_join PRIVATE ${CMAKE_SOURCE_DIR}/include $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)
target_link_libraries(test_join PRIVATE cudf_c)
target_compile_options(test_join PRIVATE -std=c11)
add_test(NAME join_compute COMMAND test_join)
set_tests_properties(join_compute PROPERTIES LABELS "gpu")
