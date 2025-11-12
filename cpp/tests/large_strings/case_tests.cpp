/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct CaseTest : public cudf::test::StringsLargeTest {};

TEST_F(CaseTest, ToLower)
{
  auto const wide = this->wide_column();
  auto input      = cudf::concatenate(std::vector<cudf::column_view>(120000, wide));  // 230MB
  auto expected   = cudf::strings::to_lower(cudf::strings_column_view(input->view()));

  int const multiplier = 12;
  std::vector<cudf::column_view> input_cols(multiplier, input->view());
  std::vector<cudf::size_type> splits;
  std::generate_n(std::back_inserter(splits), multiplier - 1, [&input, n = 1]() mutable {
    return input->view().size() * (n++);
  });

  auto large_input = cudf::concatenate(input_cols);  // 2700MB > 2GB
  auto const sv    = cudf::strings_column_view(large_input->view());
  auto result      = cudf::strings::to_lower(sv);

  // verify results in sections
  auto sliced = cudf::split(result->view(), splits);
  for (auto c : sliced) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected->view());
  }
}
