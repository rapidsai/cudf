/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/concatenate.hpp>

class ConcatenateTest : public cudf::test::BaseFixture {};

TEST_F(ConcatenateTest, Column)
{
  cudf::test::fixed_width_column_wrapper<int> const input1({0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int> const input2({1, 1, 1, 1, 1});
  std::vector<cudf::column_view> views{input1, input2};
  auto result = cudf::concatenate(views, cudf::test::get_default_stream());
}

TEST_F(ConcatenateTest, Table)
{
  cudf::test::fixed_width_column_wrapper<int> const input1({0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int> const input2({1, 1, 1, 1, 1});
  cudf::table_view tbl1({input1, input2});
  cudf::table_view tbl2({input2, input1});
  std::vector<cudf::table_view> views{tbl1, tbl2};
  auto result = cudf::concatenate(views, cudf::test::get_default_stream());
}

TEST_F(ConcatenateTest, Masks)
{
  cudf::test::fixed_width_column_wrapper<int> const input1(
    {{0, 0, 0, 0, 0}, {false, false, false, false, false}});
  cudf::test::fixed_width_column_wrapper<int> const input2(
    {{0, 0, 0, 0, 0}, {true, true, true, true, true}});
  std::vector<cudf::column_view> views{input1, input2};
  auto result = cudf::concatenate_masks(views, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
