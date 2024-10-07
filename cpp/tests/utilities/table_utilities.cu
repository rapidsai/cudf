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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

namespace cudf::test::detail {
void expect_table_properties_equal(cudf::table_view lhs, cudf::table_view rhs)
{
  EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
}

void expect_tables_equal(cudf::table_view lhs, cudf::table_view rhs)
{
  expect_table_properties_equal(lhs, rhs);
  for (auto i = 0; i < lhs.num_columns(); ++i) {
    cudf::test::detail::expect_columns_equal(lhs.column(i), rhs.column(i));
  }
}

/**
 * @copydoc cudf::test::expect_tables_equivalent
 */
void expect_tables_equivalent(cudf::table_view lhs, cudf::table_view rhs)
{
  auto num_columns = lhs.num_columns();
  for (auto i = 0; i < num_columns; ++i) {
    cudf::test::detail::expect_columns_equivalent(lhs.column(i), rhs.column(i));
  }
}

}  // namespace cudf::test::detail
