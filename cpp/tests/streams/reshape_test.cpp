/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

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
