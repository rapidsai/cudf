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
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <limits>
#include <memory>

using TestTypes = cudf::test::Types<int32_t>;

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto s = new ScalarType(cudf::test::make_type_param_scalar<T>(0), false, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  T value,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto s = new ScalarType(value, true, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T>
constexpr auto highest()
{
  // chrono types do not have std::numeric_limits specializations and should use T::max()
  // https://eel.is/c++draft/numeric.limits.general#6
  if constexpr (cudf::is_chrono<T>()) return T::max();
  return std::numeric_limits<T>::max();
}

template <typename T>
constexpr auto lowest()
{
  // chrono types do not have std::numeric_limits specializations and should use T::min()
  // https://eel.is/c++draft/numeric.limits.general#6
  if constexpr (cudf::is_chrono<T>()) return T::min();
  return std::numeric_limits<T>::lowest();
}

template <typename T>
struct ShiftTestsTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ShiftTestsTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(ShiftTestsTyped, ColumnEmpty)
{
  using T = TypeParam;

  std::vector<T> vals{};
  std::vector<bool> mask{};

  auto input    = cudf::test::fixed_width_column_wrapper<T>(vals.begin(), vals.end(), mask.begin());
  auto expected = cudf::test::fixed_width_column_wrapper<T>(vals.begin(), vals.end(), mask.begin());

  auto fill   = make_scalar<T>();
  auto actual = cudf::shift(input, 5, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTestsTyped, NonNullColumn)
{
  using T = TypeParam;

  auto input = cudf::test::fixed_width_column_wrapper<T>{lowest<T>(),
                                                         cudf::test::make_type_param_scalar<T>(1),
                                                         cudf::test::make_type_param_scalar<T>(2),
                                                         cudf::test::make_type_param_scalar<T>(3),
                                                         cudf::test::make_type_param_scalar<T>(4),
                                                         cudf::test::make_type_param_scalar<T>(5),
                                                         highest<T>()};
  auto expected =
    cudf::test::fixed_width_column_wrapper<T>{cudf::test::make_type_param_scalar<T>(7),
                                              cudf::test::make_type_param_scalar<T>(7),
                                              lowest<T>(),
                                              cudf::test::make_type_param_scalar<T>(1),
                                              cudf::test::make_type_param_scalar<T>(2),
                                              cudf::test::make_type_param_scalar<T>(3),
                                              cudf::test::make_type_param_scalar<T>(4)};

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTestsTyped, NegativeShift)
{
  using T = TypeParam;

  auto input = cudf::test::fixed_width_column_wrapper<T>{lowest<T>(),
                                                         cudf::test::make_type_param_scalar<T>(1),
                                                         cudf::test::make_type_param_scalar<T>(2),
                                                         cudf::test::make_type_param_scalar<T>(3),
                                                         cudf::test::make_type_param_scalar<T>(4),
                                                         cudf::test::make_type_param_scalar<T>(5),
                                                         highest<T>()};
  auto expected =
    cudf::test::fixed_width_column_wrapper<T>{cudf::test::make_type_param_scalar<T>(4),
                                              cudf::test::make_type_param_scalar<T>(5),
                                              highest<T>(),
                                              cudf::test::make_type_param_scalar<T>(7),
                                              cudf::test::make_type_param_scalar<T>(7),
                                              cudf::test::make_type_param_scalar<T>(7),
                                              cudf::test::make_type_param_scalar<T>(7)};

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, -4, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTestsTyped, NullScalar)
{
  using T = TypeParam;

  auto input = cudf::test::fixed_width_column_wrapper<T>{lowest<T>(),
                                                         cudf::test::make_type_param_scalar<T>(5),
                                                         cudf::test::make_type_param_scalar<T>(0),
                                                         cudf::test::make_type_param_scalar<T>(3),
                                                         cudf::test::make_type_param_scalar<T>(0),
                                                         cudf::test::make_type_param_scalar<T>(1),
                                                         highest<T>()};
  auto expected =
    cudf::test::fixed_width_column_wrapper<T>({cudf::test::make_type_param_scalar<T>(0),
                                               cudf::test::make_type_param_scalar<T>(0),
                                               lowest<T>(),
                                               cudf::test::make_type_param_scalar<T>(5),
                                               cudf::test::make_type_param_scalar<T>(0),
                                               cudf::test::make_type_param_scalar<T>(3),
                                               cudf::test::make_type_param_scalar<T>(0)},
                                              {0, 0, 1, 1, 1, 1, 1});

  auto fill = make_scalar<T>();

  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTestsTyped, NullableColumn)
{
  using T = TypeParam;

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>({1, 2, 3, 4, 5}, {0, 1, 1, 1, 0});
  auto expected =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({7, 7, 1, 2, 3}, {1, 1, 0, 1, 1});

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTestsTyped, MismatchFillValueDtypes)
{
  using T = TypeParam;

  auto input = cudf::test::fixed_width_column_wrapper<T>{};

  auto fill = cudf::string_scalar("");

  EXPECT_THROW(cudf::shift(input, 5, fill), cudf::logic_error);
}

struct ShiftTests : public cudf::test::BaseFixture {};

TEST_F(ShiftTests, StringsShiftTest)
{
  auto input =
    cudf::test::strings_column_wrapper({"", "bb", "ccc", "ddddddé", ""}, {0, 1, 1, 1, 0});

  auto fill    = cudf::string_scalar("xx");
  auto results = cudf::shift(input, 2, fill);
  auto expected_right =
    cudf::test::strings_column_wrapper({"xx", "xx", "", "bb", "ccc"}, {1, 1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right, *results);

  results = cudf::shift(input, -2, fill);
  auto expected_left =
    cudf::test::strings_column_wrapper({"ccc", "ddddddé", "", "xx", "xx"}, {1, 1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left, *results);

  auto sliced = cudf::slice(input, {1, 4}).front();

  results           = cudf::shift(sliced, 1, fill);
  auto sliced_right = cudf::test::strings_column_wrapper({"xx", "bb", "ccc"}, {1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sliced_right, *results);

  results          = cudf::shift(sliced, -1, fill);
  auto sliced_left = cudf::test::strings_column_wrapper({"ccc", "ddddddé", "xx"}, {1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sliced_left, *results);
}

TEST_F(ShiftTests, StringsShiftNullFillTest)
{
  auto input = cudf::test::strings_column_wrapper(
    {"a", "b", "c", "d", "e", "ff", "ggg", "hhhh", "iii", "jjjjj"});
  auto phil = cudf::string_scalar("", false);

  auto results  = cudf::shift(input, -1, phil);
  auto expected = cudf::test::strings_column_wrapper(
    {"b", "c", "d", "e", "ff", "ggg", "hhhh", "iii", "jjjjj", ""}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::shift(input, 1, phil);
  expected = cudf::test::strings_column_wrapper(
    {"", "a", "b", "c", "d", "e", "ff", "ggg", "hhhh", "iii"}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto sliced = cudf::slice(input, {5, 10}).front();
  results     = cudf::shift(sliced, -2, phil);
  expected = cudf::test::strings_column_wrapper({"hhhh", "iii", "jjjjj", "", ""}, {1, 1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::shift(sliced, 2, phil);
  expected = cudf::test::strings_column_wrapper({"", "", "ff", "ggg", "hhhh"}, {0, 0, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(ShiftTests, OffsetGreaterThanSize)
{
  auto const input_str =
    cudf::test::strings_column_wrapper({"", "bb", "ccc", "ddé", ""}, {0, 1, 1, 1, 0});
  auto results      = cudf::shift(input_str, 6, cudf::string_scalar("xx"));
  auto expected_str = cudf::test::strings_column_wrapper({"xx", "xx", "xx", "xx", "xx"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_str, *results);
  results = cudf::shift(input_str, -6, cudf::string_scalar("xx"));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_str, *results);

  results      = cudf::shift(input_str, 6, cudf::string_scalar("", false));
  expected_str = cudf::test::strings_column_wrapper({"", "", "", "", ""}, {0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_str, *results);
  results = cudf::shift(input_str, -6, cudf::string_scalar("", false));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_str, *results);

  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>({0, 2, 3, 4, 0}, {0, 1, 1, 1, 0});
  results       = cudf::shift(input, 6, cudf::numeric_scalar<int32_t>(9));
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({9, 9, 9, 9, 9});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  results = cudf::shift(input, -6, cudf::numeric_scalar<int32_t>(9));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);

  results  = cudf::shift(input, 6, cudf::numeric_scalar<int32_t>(0, false));
  expected = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  results = cudf::shift(input, -6, cudf::numeric_scalar<int32_t>(0, false));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
}
