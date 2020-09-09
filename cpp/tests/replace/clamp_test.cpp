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

#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <gtest/gtest.h>

struct ClampErrorTest : public cudf::test::BaseFixture {
};

TEST_F(ClampErrorTest, MisMatchingScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT64));
  hi->set_valid(true);

  cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::logic_error);
}

TEST_F(ClampErrorTest, MisMatchingInputAndScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::logic_error);
}

TEST_F(ClampErrorTest, MisMatchingReplaceScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT64));
  lo_replace->set_valid(true);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::logic_error);
}

TEST_F(ClampErrorTest, InValidCase1)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo_replace->set_valid(false);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::logic_error);
}

TEST_F(ClampErrorTest, InValidCase2)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo_replace->set_valid(true);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid(false);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::logic_error);
}

struct ClampEmptyCaseTest : public cudf::test::BaseFixture {
};

TEST_F(ClampEmptyCaseTest, BothScalarEmptyInvalid)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(false);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(false);

  cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

TEST_F(ClampEmptyCaseTest, EmptyInput)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid(true);

  cudf::test::fixed_width_column_wrapper<int32_t> input({});

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

template <class T>
struct ClampTestNumeric : public cudf::test::BaseFixture {
  std::unique_ptr<cudf::column> run_clamp(std::vector<T> input,
                                          std::vector<cudf::size_type> input_validity,
                                          T lo,
                                          bool lo_validity,
                                          T hi,
                                          bool hi_validity,
                                          T lo_replace,
                                          bool lo_replace_validity,
                                          T hi_replace,
                                          bool hi_replace_validity)
  {
    using ScalarType = cudf::scalar_type_t<T>;
    std::unique_ptr<cudf::scalar> lo_scalar{nullptr};
    std::unique_ptr<cudf::scalar> hi_scalar{nullptr};
    std::unique_ptr<cudf::scalar> lo_replace_scalar{nullptr};
    std::unique_ptr<cudf::scalar> hi_replace_scalar{nullptr};
    if (cudf::is_numeric<T>()) {
      lo_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_timestamp<T>()) {
      lo_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_duration<T>()) {
      lo_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    }

    static_cast<ScalarType*>(lo_scalar.get())->set_value(lo);
    static_cast<ScalarType*>(lo_scalar.get())->set_valid(lo_validity);
    static_cast<ScalarType*>(lo_replace_scalar.get())->set_value(lo_replace);
    static_cast<ScalarType*>(lo_replace_scalar.get())->set_valid(lo_replace_validity);
    static_cast<ScalarType*>(hi_scalar.get())->set_value(hi);
    static_cast<ScalarType*>(hi_scalar.get())->set_valid(hi_validity);
    static_cast<ScalarType*>(hi_replace_scalar.get())->set_value(hi_replace);
    static_cast<ScalarType*>(hi_replace_scalar.get())->set_valid(hi_replace_validity);

    if (input.size() == input_validity.size()) {
      cudf::test::fixed_width_column_wrapper<T> input_column(
        input.begin(), input.end(), input_validity.begin());

      return cudf::clamp(
        input_column, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
    } else {
      cudf::test::fixed_width_column_wrapper<T> input_column(input.begin(), input.end());
      return cudf::clamp(
        input_column, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
    }
  }
};
using Types = cudf::test::FixedWidthTypesWithoutFixedPoint;

TYPED_TEST_CASE(ClampTestNumeric, Types);

TYPED_TEST(ClampTestNumeric, WithNoNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, true, hi, true, lo, true, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, LowerNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, false, hi, true, lo, false, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, UpperNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, true, hi, false, lo, true, hi, false);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, InputNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::vector<cudf::size_type> input_validity({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  auto got = this->run_clamp(input, input_validity, lo, true, hi, true, lo, true, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8},
                                                              {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, InputNulliWithReplace)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  T lo_replace(cudf::test::make_type_param_scalar<T>(16));
  T hi_replace(cudf::test::make_type_param_scalar<T>(32));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::vector<cudf::size_type> input_validity({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  auto got =
    this->run_clamp(input, input_validity, lo, true, hi, true, lo_replace, true, hi_replace, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({16, 16, 2, 3, 4, 5, 6, 7, 8, 32, 32},
                                                              {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

template <typename T>
struct ClampFloatTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ClampFloatTest, cudf::test::FloatingPointTypes);

TYPED_TEST(ClampFloatTest, WithNANandNoNull)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(8.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(1.0), T(NAN), T(2.0), T(9.0)});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(2.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(6.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid(true);

  auto got = cudf::clamp(input, *lo_scalar, *hi_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(6.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(2.0), T(NAN), T(2.0), T(6.0)});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampFloatTest, WithNANandNull)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(8.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(1.0), T(NAN), T(2.0), T(9.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(2.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(6.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid(true);

  auto got = cudf::clamp(input, *lo_scalar, *hi_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(6.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(2.0), T(NAN), T(2.0), T(6.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampFloatTest, SignOfAFloat)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(2.0), T(0.0), T(NAN), T(4.0), T(-0.5), T(-1.0), T(1.0), T(NAN), T(0.5), T(9.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto lo_replace_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_replace_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(0.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(0.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid(true);
  static_cast<ScalarType*>(lo_replace_scalar.get())->set_value(-1.0);
  static_cast<ScalarType*>(lo_replace_scalar.get())->set_valid(true);
  static_cast<ScalarType*>(hi_replace_scalar.get())->set_value(1.0);
  static_cast<ScalarType*>(hi_replace_scalar.get())->set_valid(true);

  auto got = cudf::clamp(input, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(1.0), T(0.0), T(NAN), T(4.0), T(-1.0), T(-1.0), T(1.0), T(NAN), T(1.0), T(1.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

struct ClampStringTest : public cudf::test::BaseFixture {
};

TEST_F(ClampStringTest, WithNullableColumn)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid(true);
  hi->set_valid(true);

  std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "i", "e", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNonNullableColumn)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid(true);
  hi->set_valid(true);

  std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "e", "e", "B"};

  cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullLow)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid(false);
  hi->set_valid(true);

  std::vector<std::string> expected_strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "e", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullHigh)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid(true);
  hi->set_valid(false);

  std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnBothLoAndHiNull)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid(false);
  hi->set_valid(false);

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

TEST_F(ClampStringTest, WithReplaceString)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo         = cudf::make_string_scalar("B");
  auto lo_replace = cudf::make_string_scalar("Z");
  auto hi         = cudf::make_string_scalar("e");
  auto hi_replace = cudf::make_string_scalar("z");
  lo->set_valid(true);
  lo_replace->set_valid(true);
  hi->set_valid(true);
  hi_replace->set_valid(true);

  std::vector<std::string> expected_strings{"Z", "b", "c", "D", "e", "F", "G", "H", "z", "z", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

CUDF_TEST_PROGRAM_MAIN()
