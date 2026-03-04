/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>

class SearchTest : public cudf::test::BaseFixture {};

TEST_F(SearchTest, LowerBound)
{
  cudf::test::fixed_width_column_wrapper<int32_t> column{10, 20, 30, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 0, 0, 1, 2, 3, 3, 4, 4, 5};

  cudf::lower_bound({cudf::table_view{{column}}},
                    {cudf::table_view{{values}}},
                    {cudf::order::ASCENDING},
                    {cudf::null_order::BEFORE},
                    cudf::test::get_default_stream());
}

TEST_F(SearchTest, UpperBound)
{
  cudf::test::fixed_width_column_wrapper<int32_t> column{10, 20, 30, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 0, 0, 1, 2, 3, 3, 4, 4, 5};

  cudf::upper_bound({cudf::table_view{{column}}},
                    {cudf::table_view{{values}}},
                    {cudf::order::ASCENDING},
                    {cudf::null_order::BEFORE},
                    cudf::test::get_default_stream());
}

TEST_F(SearchTest, ContainsScalar)
{
  cudf::test::fixed_width_column_wrapper<int32_t> column{0, 1, 17, 19, 23, 29, 71};
  cudf::numeric_scalar<int32_t> scalar{23, true, cudf::test::get_default_stream()};

  cudf::contains(column, scalar, cudf::test::get_default_stream());
}

TEST_F(SearchTest, ContainsColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> haystack{0, 1, 17, 19, 23, 29, 71};
  cudf::test::fixed_width_column_wrapper<int32_t> needles{17, 19, 45, 72};

  cudf::test::fixed_width_column_wrapper<bool> expect{1, 1, 0, 0};

  cudf::contains(haystack, needles, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
