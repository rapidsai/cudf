/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/reshape.hpp>

class ReshapeTest : public cudf::test::BaseFixture {};

TEST_F(ReshapeTest, InterleaveColumns)
{
  auto a = cudf::test::fixed_width_column_wrapper<int32_t>({0, 3, 6});
  auto b = cudf::test::fixed_width_column_wrapper<int32_t>({1, 4, 7});
  auto c = cudf::test::fixed_width_column_wrapper<int32_t>({2, 5, 8});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});
  cudf::interleave_columns(in, cudf::test::get_default_stream());
}

TEST_F(ReshapeTest, Tile)
{
  auto a = cudf::test::fixed_width_column_wrapper<int32_t>({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  cudf::tile(in, 2, cudf::test::get_default_stream());
}

TEST_F(ReshapeTest, ByteCast)
{
  auto a = cudf::test::fixed_width_column_wrapper<int32_t>({0, 100, -100, 1000, 1000});
  cudf::byte_cast(a, cudf::flip_endianness::YES, cudf::test::get_default_stream());
  cudf::byte_cast(a, cudf::flip_endianness::NO, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
