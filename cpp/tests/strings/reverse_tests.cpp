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
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct StringsReverseTest : public cudf::test::BaseFixture {
};

TEST_F(StringsReverseTest, Reverse)
{
  cudf::test::strings_column_wrapper input({"abcdef", "12345", "", "", "aébé", "A é Z"});

  auto results = cudf::strings::reverse(cudf::strings_column_view(input));

  cudf::test::strings_column_wrapper expected({"fedcba", "54321", "", "", "ébéa", "Z é A"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReverseTest, EmptyStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::reverse(cudf::strings_column_view(zero_size_strings_column));
  auto view    = results->view();
  cudf::test::expect_column_empty(results->view());
}
