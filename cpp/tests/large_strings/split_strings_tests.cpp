/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

struct StringsSplitTest : public cudf::test::StringsLargeTest {};

TEST_F(StringsSplitTest, Split)
{
  auto const expected   = this->long_column();
  auto const view       = cudf::column_view(expected);
  auto const multiplier = 10;
  auto const separator  = cudf::string_scalar("|");
  auto const input      = cudf::strings::concatenate(
    cudf::table_view(std::vector<cudf::column_view>(multiplier, view)), separator);

  {
    auto result = cudf::strings::split(cudf::strings_column_view(input->view()), separator);
    for (auto c : result->view()) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected);
    }
  }

  auto lc = cudf::strings::split_record(cudf::strings_column_view(input->view()), separator);
  auto lv = cudf::lists_column_view(lc->view());
  auto sv = cudf::strings_column_view(lv.child());
  EXPECT_EQ(sv.size(), view.size() * multiplier);
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});
}
