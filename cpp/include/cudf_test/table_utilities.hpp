/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace test::detail {
/**
 * @brief Verifies the property equality of two tables.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL`
 * instead.
 *
 * @param lhs The first table
 * @param rhs The second table
 */
void expect_table_properties_equal(cudf::table_view lhs, cudf::table_view rhs);

/**
 * @brief Verifies the equality of two tables.
 *
 * Treats null elements as equivalent.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_TABLES_EQUAL` instead.
 *
 * @param lhs The first table
 * @param rhs The second table
 */
void expect_tables_equal(cudf::table_view lhs, cudf::table_view rhs);

/**
 * @brief Verifies the equivalency of two tables.
 *
 * Treats null elements as equivalent.  Columns that have nullability but no nulls,
 * and columns that are not nullable are considered equivalent.
 *
 * @note This function should not be used directly. Use `CUDF_TEST_EXPECT_TABLES_EQUIVALENT`
 * instead.
 *
 * @param lhs The first table
 * @param rhs The second table
 */
void expect_tables_equivalent(cudf::table_view lhs, cudf::table_view rhs);

}  // namespace test::detail
}  // namespace CUDF_EXPORT cudf

// Macros for showing line of failure.
#define CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(lhs, rhs)        \
  do {                                                           \
    SCOPED_TRACE(" <--  line of failure\n");                     \
    cudf::test::detail::expect_table_properties_equal(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_TABLES_EQUAL(lhs, rhs)        \
  do {                                                 \
    SCOPED_TRACE(" <--  line of failure\n");           \
    cudf::test::detail::expect_tables_equal(lhs, rhs); \
  } while (0)

#define CUDF_TEST_EXPECT_TABLES_EQUIVALENT(lhs, rhs)        \
  do {                                                      \
    SCOPED_TRACE(" <--  line of failure\n");                \
    cudf::test::detail::expect_tables_equivalent(lhs, rhs); \
  } while (0)
