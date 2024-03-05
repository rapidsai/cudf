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
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

class FillingTest : public cudf::test::BaseFixture {};

TEST_F(FillingTest, FillInPlace)
{
  cudf::test::fixed_width_column_wrapper<int> col({0, 0, 0, 0, 0});
  auto scalar = cudf::numeric_scalar<int>(5, true, cudf::test::get_default_stream());
  cudf::mutable_column_view mut_view = col;
  cudf::fill_in_place(mut_view, 0, 4, scalar, cudf::test::get_default_stream());
}

TEST_F(FillingTest, Fill)
{
  cudf::test::fixed_width_column_wrapper<int> const col({0, 0, 0, 0, 0});
  auto scalar = cudf::numeric_scalar<int>(5, true, cudf::test::get_default_stream());
  cudf::fill(col, 0, 4, scalar, cudf::test::get_default_stream());
}

TEST_F(FillingTest, RepeatVariable)
{
  cudf::test::fixed_width_column_wrapper<int> const col({0, 0, 0, 0, 0});
  cudf::table_view const table({col});
  cudf::test::fixed_width_column_wrapper<int> const counts({1, 2, 3, 4, 5});
  cudf::repeat(table, counts, cudf::test::get_default_stream());
}

TEST_F(FillingTest, RepeatConst)
{
  cudf::test::fixed_width_column_wrapper<int> const col({0, 0, 0, 0, 0});
  cudf::table_view const table({col});
  cudf::repeat(table, 5, cudf::test::get_default_stream());
}

TEST_F(FillingTest, SequenceStep)
{
  auto init = cudf::numeric_scalar<int>(5, true, cudf::test::get_default_stream());
  auto step = cudf::numeric_scalar<int>(2, true, cudf::test::get_default_stream());
  cudf::sequence(10, init, step, cudf::test::get_default_stream());
}

TEST_F(FillingTest, Sequence)
{
  auto init = cudf::numeric_scalar<int>(5, true, cudf::test::get_default_stream());
  cudf::sequence(10, init, cudf::test::get_default_stream());
}

TEST_F(FillingTest, CalendricalMonthSequence)
{
  cudf::timestamp_scalar<cudf::timestamp_s> init(
    1629852896L, true, cudf::test::get_default_stream());  // 2021-08-25 00:54:56 GMT

  cudf::calendrical_month_sequence(10, init, 2, cudf::test::get_default_stream());
}
