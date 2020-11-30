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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

using cudf::test::fixed_width_column_wrapper;
using TestTypes = cudf::test::Types<int32_t>;

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto s = new ScalarType(cudf::test::make_type_param_scalar<T>(0), false, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  T value,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto s = new ScalarType(value, true, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T>
auto lowest = std::numeric_limits<T>::lowest();

template <typename T>
auto highest = std::numeric_limits<T>::max();

template <typename T>
struct ShiftTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ShiftTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ShiftTest, OneColumnEmpty)
{
  using T = TypeParam;

  std::vector<T> vals{};
  std::vector<bool> mask{};

  auto input    = fixed_width_column_wrapper<T>{};
  auto expected = fixed_width_column_wrapper<T>(vals.begin(), vals.end(), mask.begin());

  auto fill   = make_scalar<T>();
  auto actual = cudf::shift(input, 5, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, TwoColumnsEmpty)
{
  using T = TypeParam;

  std::vector<T> vals{};
  std::vector<bool> mask{};

  auto input    = fixed_width_column_wrapper<T>(vals.begin(), vals.end(), mask.begin());
  auto expected = fixed_width_column_wrapper<T>(vals.begin(), vals.end(), mask.begin());

  auto fill   = make_scalar<T>();
  auto actual = cudf::shift(input, 5, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, OneColumn)
{
  using T = TypeParam;

  auto input    = fixed_width_column_wrapper<T>{lowest<T>,
                                             cudf::test::make_type_param_scalar<T>(1),
                                             cudf::test::make_type_param_scalar<T>(2),
                                             cudf::test::make_type_param_scalar<T>(3),
                                             cudf::test::make_type_param_scalar<T>(4),
                                             cudf::test::make_type_param_scalar<T>(5),
                                             highest<T>};
  auto expected = fixed_width_column_wrapper<T>{cudf::test::make_type_param_scalar<T>(7),
                                                cudf::test::make_type_param_scalar<T>(7),
                                                lowest<T>,
                                                cudf::test::make_type_param_scalar<T>(1),
                                                cudf::test::make_type_param_scalar<T>(2),
                                                cudf::test::make_type_param_scalar<T>(3),
                                                cudf::test::make_type_param_scalar<T>(4)};

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, OneColumnNegativeShift)
{
  using T = TypeParam;

  auto input    = fixed_width_column_wrapper<T>{lowest<T>,
                                             cudf::test::make_type_param_scalar<T>(1),
                                             cudf::test::make_type_param_scalar<T>(2),
                                             cudf::test::make_type_param_scalar<T>(3),
                                             cudf::test::make_type_param_scalar<T>(4),
                                             cudf::test::make_type_param_scalar<T>(5),
                                             highest<T>};
  auto expected = fixed_width_column_wrapper<T>{cudf::test::make_type_param_scalar<T>(4),
                                                cudf::test::make_type_param_scalar<T>(5),
                                                highest<T>,
                                                cudf::test::make_type_param_scalar<T>(7),
                                                cudf::test::make_type_param_scalar<T>(7),
                                                cudf::test::make_type_param_scalar<T>(7),
                                                cudf::test::make_type_param_scalar<T>(7)};

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, -4, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, OneColumnNullFill)
{
  using T = TypeParam;

  auto input    = fixed_width_column_wrapper<T>{lowest<T>,
                                             cudf::test::make_type_param_scalar<T>(5),
                                             cudf::test::make_type_param_scalar<T>(0),
                                             cudf::test::make_type_param_scalar<T>(3),
                                             cudf::test::make_type_param_scalar<T>(0),
                                             cudf::test::make_type_param_scalar<T>(1),
                                             highest<T>};
  auto expected = fixed_width_column_wrapper<T>({cudf::test::make_type_param_scalar<T>(0),
                                                 cudf::test::make_type_param_scalar<T>(0),
                                                 lowest<T>,
                                                 cudf::test::make_type_param_scalar<T>(5),
                                                 cudf::test::make_type_param_scalar<T>(0),
                                                 cudf::test::make_type_param_scalar<T>(3),
                                                 cudf::test::make_type_param_scalar<T>(0)},
                                                {0, 0, 1, 1, 1, 1, 1});

  auto fill = make_scalar<T>();

  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, TwoColumnsNullableInput)
{
  using T = TypeParam;

  auto input    = fixed_width_column_wrapper<T, int32_t>({1, 2, 3, 4, 5}, {0, 1, 1, 1, 0});
  auto expected = fixed_width_column_wrapper<T, int32_t>({7, 7, 1, 2, 3}, {1, 1, 0, 1, 1});

  auto fill   = make_scalar<T>(cudf::test::make_type_param_scalar<T>(7));
  auto actual = cudf::shift(input, 2, *fill);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TYPED_TEST(ShiftTest, MismatchFillValueDtypes)
{
  using T = TypeParam;

  if (std::is_same<T, int>::value) { return; }

  auto input = fixed_width_column_wrapper<T>{};

  auto fill = make_scalar<int>();

  std::unique_ptr<cudf::column> output;

  EXPECT_THROW(output = cudf::shift(input, 5, *fill), cudf::logic_error);
}
