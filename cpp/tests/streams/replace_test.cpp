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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>

class ReplaceTest : public cudf::test::BaseFixture {};

TEST_F(ReplaceTest, ReplaceNullsColumn)
{
  cudf::test::fixed_width_column_wrapper<int> input(
    {{0, 0, 0, 0, 0}, {false, false, true, true, true}});
  cudf::test::fixed_width_column_wrapper<int> replacement({1, 1, 1, 1, 1});
  cudf::replace_nulls(input, replacement, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, ReplaceNullsScalar)
{
  cudf::test::fixed_width_column_wrapper<int> input(
    {{0, 0, 0, 0, 0}, {false, false, true, true, true}});
  auto replacement = cudf::numeric_scalar<int>(1, true, cudf::test::get_default_stream());
  cudf::replace_nulls(input, replacement, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, ReplaceNullsPolicy)
{
  cudf::test::fixed_width_column_wrapper<int> input(
    {{0, 0, 0, 0, 0}, {false, false, true, true, true}});
  cudf::replace_nulls(input, cudf::replace_policy::FOLLOWING, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, ReplaceNansColumn)
{
  auto nan          = std::numeric_limits<double>::quiet_NaN();
  auto input_column = cudf::test::make_type_param_vector<double>({0.0, 0.0, nan, nan, nan});
  cudf::test::fixed_width_column_wrapper<double> input(input_column.begin(), input_column.end());
  cudf::test::fixed_width_column_wrapper<double> replacement({0, 1, 2, 3, 4});
  cudf::replace_nans(input, replacement, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, ReplaceNansScalar)
{
  auto nan          = std::numeric_limits<double>::quiet_NaN();
  auto input_column = cudf::test::make_type_param_vector<double>({0.0, 0.0, nan, nan, nan});
  cudf::test::fixed_width_column_wrapper<double> input(input_column.begin(), input_column.end());
  auto replacement = cudf::numeric_scalar<double>(4, true, cudf::test::get_default_stream());
  cudf::replace_nans(input, replacement, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, FindAndReplaceAll)
{
  cudf::test::fixed_width_column_wrapper<int> input({0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int> values_to_replace({0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int> replacement_values({1, 1, 1, 1, 1});
  cudf::find_and_replace_all(
    input, values_to_replace, replacement_values, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, ClampWithReplace)
{
  cudf::test::fixed_width_column_wrapper<int> input({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto low          = cudf::numeric_scalar<int>(3, true, cudf::test::get_default_stream());
  auto low_replace  = cudf::numeric_scalar<int>(5, true, cudf::test::get_default_stream());
  auto high         = cudf::numeric_scalar<int>(7, true, cudf::test::get_default_stream());
  auto high_replace = cudf::numeric_scalar<int>(6, true, cudf::test::get_default_stream());
  cudf::clamp(input, low, low_replace, high, high_replace, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, Clamp)
{
  cudf::test::fixed_width_column_wrapper<int> input({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto low  = cudf::numeric_scalar<int>(3, true, cudf::test::get_default_stream());
  auto high = cudf::numeric_scalar<int>(7, true, cudf::test::get_default_stream());
  cudf::clamp(input, low, high, cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, NormalizeNansAndZeros)
{
  auto nan          = std::numeric_limits<double>::quiet_NaN();
  auto input_column = cudf::test::make_type_param_vector<double>({-0.0, 0.0, -nan, nan, nan});
  cudf::test::fixed_width_column_wrapper<double> input(input_column.begin(), input_column.end());
  cudf::normalize_nans_and_zeros(static_cast<cudf::column_view>(input),
                                 cudf::test::get_default_stream());
}

TEST_F(ReplaceTest, NormalizeNansAndZerosMutable)
{
  auto nan          = std::numeric_limits<double>::quiet_NaN();
  auto input_column = cudf::test::make_type_param_vector<double>({-0.0, 0.0, -nan, nan, nan});
  cudf::test::fixed_width_column_wrapper<double> input(input_column.begin(), input_column.end());
  cudf::mutable_column_view mutable_view = cudf::column(input, cudf::test::get_default_stream());
  cudf::normalize_nans_and_zeros(mutable_view, cudf::test::get_default_stream());
}
