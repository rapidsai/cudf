/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/merge.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct MergeTest : public cudf::test::StringsLargeTest {};

TEST_F(MergeTest, MergeLargeStrings)
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
  auto const column_order    = std::vector<cudf::order>{cudf::order::ASCENDING};
  auto const null_precedence = std::vector<cudf::null_order>{cudf::null_order::AFTER};

  auto result = cudf::merge(input_views, {0}, column_order, null_precedence);
  auto sv     = cudf::strings_column_view(result->view().column(0));
  EXPECT_EQ(sv.size(), view.num_rows() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  auto sliced = cudf::split(sv.parent(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, input);
  }

  // also test with large strings column as input
  input_views.clear();
  input_views.push_back(view);            // regular column
  input_views.push_back(result->view());  // large column
  result = cudf::merge(input_views, {0}, column_order, null_precedence);
  sv     = cudf::strings_column_view(result->view().column(0));
  EXPECT_EQ(sv.size(), view.num_rows() * (multiplier + 1));
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});
  splits.push_back(view.num_rows() * multiplier);
  sliced = cudf::split(sv.parent(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, input);
  }

  // also check merge still returns 32-bit offsets for regular columns
  input_views.clear();
  input_views.push_back(view);
  input_views.push_back(view);
  result = cudf::merge(input_views, {0}, column_order, null_precedence);
  sv     = cudf::strings_column_view(result->view().column(0));
  EXPECT_EQ(sv.size(), view.num_rows() * 2);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT32});
  sliced = cudf::split(sv.parent(), {view.num_rows()});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced[0], input);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced[1], input);
}
