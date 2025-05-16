/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct StringsReverseTest : public cudf::test::BaseFixture {};

TEST_F(StringsReverseTest, Reverse)
{
  auto input =
    cudf::test::strings_column_wrapper({"abcdef", "12345", "", "", "aébé", "A é Z", "X", "é"},
                                       {true, true, true, false, true, true, true, true});
  auto results = cudf::strings::reverse(cudf::strings_column_view(input));
  auto expected =
    cudf::test::strings_column_wrapper({"fedcba", "54321", "", "", "ébéa", "Z é A", "X", "é"},
                                       {true, true, true, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  auto sliced = cudf::slice(input, {1, 7}).front();
  results     = cudf::strings::reverse(cudf::strings_column_view(sliced));
  expected    = cudf::test::strings_column_wrapper({"54321", "", "", "ébéa", "Z é A", "X"},
                                                   {true, true, false, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReverseTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto results = cudf::strings::reverse(cudf::strings_column_view(zero_size_strings_column));
  auto view    = results->view();
  cudf::test::expect_column_empty(results->view());
}
