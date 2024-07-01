/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

/**
 * @brief test macro to be expects `expr` to return cudaSuccess
 *
 * This will stop the test process on failure.
 *
 * @param expr expression to be tested
 */
#define ASSERT_CUDA_SUCCEEDED(expr) ASSERT_EQ(cudaSuccess, expr)
/**
 * @brief test macro to be expects `expr` to return cudaSuccess
 *
 * @param expr expression to be tested
 */
#define EXPECT_CUDA_SUCCEEDED(expr) EXPECT_EQ(cudaSuccess, expr)

/**
 * @brief test macro to be expected as no exception.
 *
 * The testing is same with EXPECT_NO_THROW() in gtest.
 * It also outputs captured error message, useful for debugging.
 *
 * @param statement The statement to be tested
 */
#define CUDF_EXPECT_NO_THROW(statement)                                                       \
  try {                                                                                       \
    statement;                                                                                \
  } catch (std::exception & e) {                                                              \
    FAIL() << "statement:" << #statement << std::endl << "reason: " << e.what() << std::endl; \
  }

/**
 * @brief test macro comparing for equality of \p lhs and \p rhs for the first \p size elements.
 */
#define CUDF_TEST_EXPECT_VECTOR_EQUAL(lhs, rhs, size)          \
  do {                                                         \
    for (decltype(size) i = 0; i < size; i++)                  \
      EXPECT_EQ(lhs[i], rhs[i]) << "Mismatch at index #" << i; \
  } while (0)
