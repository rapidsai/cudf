/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/reverse.hpp>

#include <vector>

using ints_lists = cudf::test::lists_column_wrapper<int32_t>;

struct ListsReverseTest : public cudf::test::BaseFixture {
};

TEST_F(ListsReverseTest, SimpleReverse)
{
  auto const input    = ints_lists{{}, {1, 2, 3}, {}, {4, 5}};
  auto const expected = ints_lists{{}, {3, 2, 1}, {}, {5, 4}};
  auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
}
