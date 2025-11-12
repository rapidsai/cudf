/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
