/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//TODO after legacy are removed
//#include "cudf_gtest.hpp"
//#include <gmock/gmock.h>

#define ASSERT_CUDA_SUCCEEDED(expr) ASSERT_EQ(cudaSuccess, expr)
#define EXPECT_CUDA_SUCCEEDED(expr) EXPECT_EQ(cudaSuccess, expr)

#define ASSERT_CUDF_SUCCEEDED(gdf_error_expression)                           \
do {                                                                          \
    gdf_error _assert_cudf_success_eval_result;                               \
    ASSERT_NO_THROW(_assert_cudf_success_eval_result = gdf_error_expression); \
    const char* _assertion_failure_message = #gdf_error_expression;           \
    ASSERT_EQ(_assert_cudf_success_eval_result, GDF_SUCCESS) <<               \
      "Failing expression: " << _assertion_failure_message;                   \
} while (0)

// Utility for testing the expectation that an expression x throws the specified
// exception whose what() message ends with the msg
#define EXPECT_THROW_MESSAGE(x, exception, startswith, endswith)     \
do { \
  EXPECT_THROW({                                                     \
    try { x; }                                                       \
    catch (const exception &e) {                                     \
    ASSERT_NE(nullptr, e.what());                                    \
    EXPECT_THAT(e.what(), testing::StartsWith((startswith)));        \
    EXPECT_THAT(e.what(), testing::EndsWith((endswith)));            \
    throw;                                                           \
  }}, exception);                                                    \
} while (0)

#define CUDF_EXPECT_THROW_MESSAGE(x, msg) \
EXPECT_THROW_MESSAGE(x, cudf::logic_error, "cuDF failure at:", msg)

#define CUDA_EXPECT_THROW_MESSAGE(x, msg) \
EXPECT_THROW_MESSAGE(x, cudf::cuda_error, "CUDA error encountered at:", msg)

/**---------------------------------------------------------------------------*
 * @brief test macro to be expected as no exception.
 * The testing is same with EXPECT_NO_THROW() in gtest.
 * It also outputs captured error message, useful for debugging.
 *
 * @param statement The statement to be tested
 *---------------------------------------------------------------------------**/
#define CUDF_EXPECT_NO_THROW(statement)                 \
try{ statement; } catch (std::exception& e)             \
    { FAIL() << "statement:" << #statement << std::endl \
             << "reason: " << e.what() << std::endl; }
