/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

struct StringsManyTest : public cudf::test::StringsLargeTest {};

TEST_F(StringsManyTest, Replace)
{
  auto const expected  = this->very_long_column();
  auto const view      = cudf::column_view(expected);
  int const multiplier = 16;
  std::vector<cudf::column_view> input_cols(multiplier, view);
  std::vector<cudf::size_type> splits;
  std::generate_n(std::back_inserter(splits), multiplier - 1, [view, n = 1]() mutable {
    return view.size() * (n++);
  });

  auto large_input = cudf::concatenate(input_cols);  // 480 million rows
  auto const sv    = cudf::strings_column_view(large_input->view());
  EXPECT_EQ(sv.size(), view.size() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  // Using replace tests reading large strings as well as creating large strings
  auto const target = cudf::string_scalar("3");  // fake the actual replace;
  auto const repl   = cudf::string_scalar("3");  // logic still builds the output
  auto result       = cudf::strings::replace(sv, target, repl);

  // verify results in sections
  auto sliced = cudf::split(result->view(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected);
  }
}
