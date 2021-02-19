/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/strings/utilities.h>

#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {
};

template <typename T>
class StringsFixedPointConvertTest : public StringsConvertTest {
};

TYPED_TEST_CASE(StringsFixedPointConvertTest, cudf::test::FixedPointTypes);

TYPED_TEST(StringsFixedPointConvertTest, ToFixedPoint)
{
  using DecimalType = TypeParam;
  using RepType     = cudf::device_storage_type_t<DecimalType>;
  using fp_wrapper  = cudf::test::fixed_point_column_wrapper<RepType>;

  cudf::test::strings_column_wrapper strings(
    {"1234", "-876", "543.2", "-0.12", ".25", "-.002", "-.0027", "", "-0.0"});
  auto results = cudf::strings::to_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_to_id<DecimalType>(), numeric::scale_type{-3}});
  auto const expected =
    fp_wrapper{{1234000, -876000, 543200, -120, 250, -2, -2, 0, 0}, numeric::scale_type{-3}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::to_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_to_id<DecimalType>(), numeric::scale_type{2}});
  auto const expected_scaled = fp_wrapper{{12, -8, 5, 0, 0, 0, 0, 0, 0}, numeric::scale_type{2}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_scaled);

  cudf::test::strings_column_wrapper strings_nulls(
    {"1234", "-876", "543", "900000", "2500000", "", ""}, {1, 1, 1, 1, 1, 1, 0});
  results = cudf::strings::to_fixed_point(cudf::strings_column_view(strings_nulls),
                                          cudf::data_type{cudf::type_to_id<DecimalType>()});
  auto const expected_nulls = fp_wrapper{
    {1234, -876, 543, 900000, 2500000, 0, 0}, {1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_nulls);
}

TYPED_TEST(StringsFixedPointConvertTest, ToFixedPointVeryLarge)
{
  using DecimalType  = TypeParam;
  using RepType      = cudf::device_storage_type_t<DecimalType>;
  using fp_wrapper   = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const strings = cudf::test::strings_column_wrapper({"1234000000000000000000",
                                                           "-876000000000000000000",
                                                           "543200000000000000000",
                                                           "-120000000000000000",
                                                           "250000000000000000",
                                                           "-2800000000000000",
                                                           "",
                                                           "-0.0"});
  auto const results = cudf::strings::to_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_to_id<DecimalType>(), numeric::scale_type{20}});
  auto const expected = fp_wrapper{{12, -8, 5, 0, 0, 0, 0, 0}, numeric::scale_type{20}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TYPED_TEST(StringsFixedPointConvertTest, ToFixedPointVerySmall)
{
  using DecimalType  = TypeParam;
  using RepType      = cudf::device_storage_type_t<DecimalType>;
  using fp_wrapper   = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const strings = cudf::test::strings_column_wrapper({"0.00000000000000001234",
                                                           "-0.0000000000000000876",
                                                           ".000000000000000005432",
                                                           "-.000000000000000012",
                                                           "+.000000000000000025",
                                                           "-.00000000002147483647",
                                                           "",
                                                           "+0.0"});
  auto const results = cudf::strings::to_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_to_id<DecimalType>(), numeric::scale_type{-20}});
  auto const expected =
    fp_wrapper{{1234, -8760, 543, -1200, 2500, -2147483647, 0, 0}, numeric::scale_type{-20}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TYPED_TEST(StringsFixedPointConvertTest, FromFixedPoint)
{
  using DecimalType = TypeParam;
  using RepType     = cudf::device_storage_type_t<DecimalType>;
  using fp_wrapper  = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const negative_scale = fp_wrapper{{110, 222, 3330, 4444, -550, -6}, numeric::scale_type{-2}};
  auto results              = cudf::strings::from_fixed_point(negative_scale);
  cudf::test::strings_column_wrapper negative_expected(
    {"1.10", "2.22", "33.30", "44.44", "-5.50", "-0.06"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, negative_expected);

  auto const positive_scale =
    fp_wrapper({110, -222, 3330, 4, -550, 0}, {1, 1, 1, 1, 1, 0}, numeric::scale_type{2});
  results = cudf::strings::from_fixed_point(positive_scale);
  cudf::test::strings_column_wrapper positive_expected(
    {"11000", "-22200", "333000", "400", "-55000", ""}, {1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, positive_expected);

  auto const zero_scale =
    fp_wrapper({0, -222, 3330, 4, -550, 0}, {0, 1, 1, 1, 1, 1}, numeric::scale_type{0});
  results = cudf::strings::from_fixed_point(zero_scale);
  cudf::test::strings_column_wrapper zero_expected({"", "-222", "3330", "4", "-550", "0"},
                                                   {0, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, zero_expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumnFixedPoint)
{
  auto zero_size_column = cudf::make_empty_column(cudf::data_type{cudf::type_id::DECIMAL32});

  auto results = cudf::strings::from_fixed_point(zero_size_column->view());
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeFixedPointColumn)
{
  auto zero_size_column = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto results = cudf::strings::to_fixed_point(zero_size_column->view(),
                                               cudf::data_type{cudf::type_id::DECIMAL32});
  EXPECT_EQ(0, results->size());
  results = cudf::strings::is_fixed_point(zero_size_column->view());
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, FromToFixedPointError)
{
  auto dtype  = cudf::data_type{cudf::type_id::INT32};
  auto column = cudf::make_numeric_column(dtype, 100);
  EXPECT_THROW(cudf::strings::from_fixed_point(column->view()), cudf::logic_error);

  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  cudf::strings_column_view strings_view(strings);
  EXPECT_THROW(cudf::strings::to_fixed_point(strings_view, dtype), cudf::logic_error);
  EXPECT_THROW(cudf::strings::is_fixed_point(strings_view, dtype), cudf::logic_error);
}

TEST_F(StringsConvertTest, IsFixedPoint)
{
  cudf::test::strings_column_wrapper strings(
    {"1234", "+876", "543.2", "-00.120", "1E34", "1.0.02", "", "-0.0"});
  auto results        = cudf::strings::is_fixed_point(cudf::strings_column_view(strings));
  auto const expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, true, true, true, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::is_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_id::DECIMAL32, numeric::scale_type{-1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::is_fixed_point(
    cudf::strings_column_view(strings),
    cudf::data_type{cudf::type_id::DECIMAL32, numeric::scale_type{1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  cudf::test::strings_column_wrapper big_numbers({"2147483647",
                                                  "-2147483647",
                                                  "2147483648",
                                                  "9223372036854775807",
                                                  "-9223372036854775807",
                                                  "9223372036854775808"});
  results = cudf::strings::is_fixed_point(cudf::strings_column_view(big_numbers),
                                          cudf::data_type{cudf::type_id::DECIMAL32});
  auto const expected32 =
    cudf::test::fixed_width_column_wrapper<bool>({true, true, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected32);

  results = cudf::strings::is_fixed_point(cudf::strings_column_view(big_numbers),
                                          cudf::data_type{cudf::type_id::DECIMAL64});
  auto const expected64 =
    cudf::test::fixed_width_column_wrapper<bool>({true, true, true, true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected64);

  results = cudf::strings::is_fixed_point(
    cudf::strings_column_view(big_numbers),
    cudf::data_type{cudf::type_id::DECIMAL32, numeric::scale_type{10}});
  auto const expected32_scaled =
    cudf::test::fixed_width_column_wrapper<bool>({true, true, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected32_scaled);

  results = cudf::strings::is_fixed_point(
    cudf::strings_column_view(big_numbers),
    cudf::data_type{cudf::type_id::DECIMAL64, numeric::scale_type{-5}});
  auto const expected64_scaled =
    cudf::test::fixed_width_column_wrapper<bool>({true, true, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected64_scaled);
}
