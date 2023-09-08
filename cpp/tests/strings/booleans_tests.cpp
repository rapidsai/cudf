/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, ToBooleans)
{
  std::vector<char const*> h_strings{"false", nullptr, "", "true", "True", "False"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_booleans(strings_view);

  std::vector<bool> h_expected{false, false, false, true, false, false};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, FromBooleans)
{
  std::vector<char const*> h_strings{"true", nullptr, "false", "true", "true", "false"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<bool> h_column{true, false, false, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> column(
    h_column.begin(),
    h_column.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_booleans(column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumnBoolean)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::BOOL8)->view();
  auto results                = cudf::strings::from_booleans(zero_size_column);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeBooleansColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto results                        = cudf::strings::to_booleans(zero_size_strings_column);
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, BooleanError)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 100);
  EXPECT_THROW(cudf::strings::from_booleans(column->view()), cudf::logic_error);
}
