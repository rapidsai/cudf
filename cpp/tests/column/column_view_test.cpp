/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <random>

template <typename T, typename T2 = void>
struct rep_type_impl {
  using type = void;
};

template <typename T>
struct rep_type_impl<T, std::enable_if_t<cudf::is_timestamp<T>()>> {
  using type = typename T::duration::rep;
};

template <typename T>
struct rep_type_impl<T, std::enable_if_t<cudf::is_duration<T>()>> {
  using type = typename T::rep;
};

template <typename T>
struct rep_type_impl<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
  using type = typename T::rep;
};

template <typename T>
using rep_type_t = typename rep_type_impl<T>::type;

template <typename T>
struct ColumnViewAllTypesTests : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ColumnViewAllTypesTests, cudf::test::FixedWidthTypes);

template <typename FromType, typename ToType, typename Iterator>
void do_logical_cast(cudf::column_view const& column_view, Iterator begin, Iterator end)
{
  auto mutable_column_view = reinterpret_cast<cudf::mutable_column_view const&>(column_view);
  if (std::is_same<FromType, ToType>::value) {
    // Cast to same type
    auto output  = cudf::logical_cast(column_view, column_view.type());
    auto output1 = cudf::logical_cast(mutable_column_view, mutable_column_view.type());
    cudf::test::expect_columns_equal(output, column_view);
    cudf::test::expect_columns_equal(output1, mutable_column_view);
  } else if (std::is_same<rep_type_t<FromType>, ToType>::value ||
             std::is_same<FromType, rep_type_t<ToType>>::value) {
    // Cast integer to timestamp or vice versa
    cudf::data_type type{cudf::type_to_id<ToType>()};
    auto output  = cudf::logical_cast(column_view, type);
    auto output1 = cudf::logical_cast(mutable_column_view, type);
    cudf::test::fixed_width_column_wrapper<ToType, cudf::size_type> expected(begin, end);
    cudf::test::expect_columns_equal(output, expected);
    cudf::test::expect_columns_equal(output1, expected);
  } else {
    // Other casts not allowed
    cudf::data_type type{cudf::type_to_id<ToType>()};
    EXPECT_THROW(cudf::logical_cast(column_view, type), cudf::logic_error);
    EXPECT_THROW(cudf::logical_cast(mutable_column_view, type), cudf::logic_error);
  }
}

TYPED_TEST(ColumnViewAllTypesTests, LogicalCast)
{
  auto begin = thrust::make_counting_iterator(1);
  auto end   = thrust::make_counting_iterator(16);

  cudf::test::fixed_width_column_wrapper<TypeParam, cudf::size_type> input(begin, end);

  do_logical_cast<TypeParam, int8_t>(input, begin, end);
  do_logical_cast<TypeParam, int16_t>(input, begin, end);
  do_logical_cast<TypeParam, int32_t>(input, begin, end);
  do_logical_cast<TypeParam, int64_t>(input, begin, end);
  do_logical_cast<TypeParam, float>(input, begin, end);
  do_logical_cast<TypeParam, double>(input, begin, end);
  do_logical_cast<TypeParam, bool>(input, begin, end);
  do_logical_cast<TypeParam, cudf::duration_D>(input, begin, end);
  do_logical_cast<TypeParam, cudf::duration_s>(input, begin, end);
  do_logical_cast<TypeParam, cudf::duration_ms>(input, begin, end);
  do_logical_cast<TypeParam, cudf::duration_us>(input, begin, end);
  do_logical_cast<TypeParam, cudf::duration_ns>(input, begin, end);
  do_logical_cast<TypeParam, cudf::timestamp_D>(input, begin, end);
  do_logical_cast<TypeParam, cudf::timestamp_s>(input, begin, end);
  do_logical_cast<TypeParam, cudf::timestamp_ms>(input, begin, end);
  do_logical_cast<TypeParam, cudf::timestamp_us>(input, begin, end);
  do_logical_cast<TypeParam, cudf::timestamp_ns>(input, begin, end);
}
