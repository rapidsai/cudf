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

#include <cudf/copying.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

struct ReshapeTest : public cudf::test::StringsLargeTest {};

TEST_F(ReshapeTest, InterleaveLargeStrings)
{
  auto const input = this->long_column();
  auto input_views = std::vector<cudf::table_view>();
  auto const view  = cudf::table_view({input});
  std::vector<cudf::size_type> splits;
  int const multiplier = 10;
  for (int i = 0; i < multiplier; ++i) {  // 2500MB > 2GB
    input_views.push_back(view);
    splits.push_back(view.num_rows() * (i + 1));
  }
  splits.pop_back();  // remove last entry

  auto result = cudf::interleave_columns(input_views);
  auto sv     = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.size(), view.num_rows() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  auto sliced = cudf::split(sv.parent(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, input);
  }

  // also check regular sizes returns 32-bit offsets
  input_views.clear();
  input_views.push_back(view);
  input_views.push_back(view);
  result = cudf::interleave_columns(input_views);
  sv     = cudf::strings_column_view(result->view());
  EXPECT_EQ(sv.size(), view.num_rows() * 2);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT32});
  sliced = cudf::split(sv.parent(), {view.num_rows()});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced[0], input);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced[1], input);
}
