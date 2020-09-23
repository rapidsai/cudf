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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace test {
/**
 * @brief Verifies the property equality of two tables.
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
 * @param lhs The first table
 * @param rhs The second table
 */
void expect_tables_equal(cudf::table_view lhs, cudf::table_view rhs);

/*
 * @brief Verifies the equivalency of two tables.
 *
 * Treats null elements as equivalent.  Columns that have nullability but no nulls,
 * and columns that are not nullable are considered equivalent.
 *
 * @param lhs The first table
 * @param rhs The second table
 **/
void expect_tables_equivalent(cudf::table_view lhs, cudf::table_view rhs);

}  // namespace test
}  // namespace cudf

// Macros for showing line of failure.
#define CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(lhs, rhs) \
  do {                                                    \
    SCOPED_TRACE(" <--  line of failure\n");              \
    cudf::test::expect_table_properties_equal(lhs, rhs);  \
  } while (0)

#define CUDF_TEST_EXPECT_TABLES_EQUAL(lhs, rhs) \
  do {                                          \
    SCOPED_TRACE(" <--  line of failure\n");    \
    cudf::test::expect_tables_equal(lhs, rhs);  \
  } while (0)

#define CUDF_TEST_EXPECT_TABLES_EQUIVALENT(lhs, rhs) \
  do {                                               \
    SCOPED_TRACE(" <--  line of failure\n");         \
    cudf::test::expect_tables_equivalent(lhs, rhs);  \
  } while (0)
