/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
