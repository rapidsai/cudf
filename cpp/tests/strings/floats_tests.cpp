/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {
};

TEST_F(StringsConvertTest, ToFloats32)
{
  std::vector<const char*> h_strings{"1234",
                                     nullptr,
                                     "-876",
                                     "543.2",
                                     "-0.12",
                                     ".25",
                                     "-.002",
                                     "",
                                     "-0.0",
                                     "1.2e4",
                                     "NaN",
                                     "abc123",
                                     "123abc",
                                     "456e",
                                     "-1.78e+5",
                                     "-122.33644782123456789",
                                     "12e+309"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  float nanval = std::numeric_limits<float>::quiet_NaN();
  float infval = std::numeric_limits<float>::infinity();
  std::vector<float> h_expected{1234.0,
                                0,
                                -876.0,
                                543.2,
                                -0.12,
                                0.25,
                                -0.002,
                                0,
                                -0.0,
                                12000,
                                nanval,
                                0,
                                123.0,
                                456.0,
                                -178000.0,
                                -122.3364486694336,
                                infval};

  auto strings_view = cudf::strings_column_view(strings);
  auto results = cudf::strings::to_floats(strings_view, cudf::data_type{cudf::type_id::FLOAT32});

  cudf::test::fixed_width_column_wrapper<float> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equivalent(*results, expected);
}

TEST_F(StringsConvertTest, FromFloats32)
{
  std::vector<float> h_floats{100,
                              654321.25,
                              -12761.125,
                              0,
                              5,
                              -4,
                              std::numeric_limits<float>::quiet_NaN(),
                              839542223232.79,
                              -0.0};
  std::vector<const char*> h_expected{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "8.395422433e+11", "-0.0"};

  cudf::test::fixed_width_column_wrapper<float> floats(
    h_floats.begin(),
    h_floats.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_floats(floats);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::expect_columns_equivalent(*results, expected);
}

TEST_F(StringsConvertTest, ToFloats64)
{
  std::vector<const char*> h_strings{"1234",
                                     nullptr,
                                     "-876",
                                     "543.2",
                                     "-0.12",
                                     ".25",
                                     "-.002",
                                     "",
                                     "-0.0",
                                     "1.28e256",
                                     "NaN",
                                     "abc123",
                                     "123abc",
                                     "456e",
                                     "-1.78e+5",
                                     "-122.33644782",
                                     "12e+309"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  double nanval = std::numeric_limits<double>::quiet_NaN();
  double infval = std::numeric_limits<double>::infinity();
  std::vector<double> h_expected{1234.0,
                                 0,
                                 -876.0,
                                 543.2,
                                 -0.12,
                                 0.25,
                                 -0.002,
                                 0,
                                 -0.0,
                                 1.28e256,
                                 nanval,
                                 0,
                                 123.0,
                                 456.0,
                                 -178000.0,
                                 -122.33644781999999,
                                 infval};

  auto strings_view = cudf::strings_column_view(strings);
  auto results = cudf::strings::to_floats(strings_view, cudf::data_type{cudf::type_id::FLOAT64});

  cudf::test::fixed_width_column_wrapper<double> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equivalent(*results, expected);
}

TEST_F(StringsConvertTest, FromFloats64)
{
  std::vector<double> h_floats{100,
                               654321.25,
                               -12761.125,
                               0,
                               5,
                               -4,
                               std::numeric_limits<double>::quiet_NaN(),
                               839542223232.794248339,
                               -0.0};
  std::vector<const char*> h_expected{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "8.395422232e+11", "-0.0"};

  cudf::test::fixed_width_column_wrapper<double> floats(
    h_floats.begin(),
    h_floats.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_floats(floats);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::expect_columns_equivalent(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumnFloat)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::type_id::FLOAT32}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::from_floats(zero_size_column);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeFloatsColumn)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results =
    cudf::strings::to_floats(zero_size_column, cudf::data_type{cudf::type_id::FLOAT32});
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, FromToFloatsError)
{
  auto dtype  = cudf::data_type{cudf::type_id::INT32};
  auto column = cudf::make_numeric_column(dtype, 100);
  EXPECT_THROW(cudf::strings::from_floats(column->view()), cudf::logic_error);

  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  EXPECT_THROW(cudf::strings::to_floats(column->view(), dtype), cudf::logic_error);
}
