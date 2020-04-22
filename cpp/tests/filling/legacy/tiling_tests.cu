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

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/legacy/filling.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/column_wrapper_factory.hpp>
#include <tests/utilities/legacy/scalar_wrapper.cuh>

#include <numeric>
#include <random>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct TileTest : GdfTest {};

using test_types = ::testing::
  Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(TileTest, test_types);

TYPED_TEST(TileTest, Tile) {
  using T = TypeParam;

  cudf::size_type repeat_count = 4;
  cudf::size_type num_values   = 9;

  // set your expectations
  cudf::size_type column_size = num_values * repeat_count;
  std::vector<int> expect_vals;
  for (cudf::size_type i = 0; i < repeat_count; i++) {
    for (cudf::size_type j = 0; j < num_values; j++) { expect_vals.push_back(2 * j); }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(
    num_values, [&](cudf::size_type i) { return 2 * i; }, [&](cudf::size_type i) { return i % 2; });

  column_wrapper<T> expected = factory.make(
    column_size,
    [&](cudf::size_type row) { return expect_vals[row]; },
    [&](cudf::size_type i) { return (i % num_values) % 2; });

  cudf::table result = cudf::tile({values.get()}, repeat_count);
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str() << "Expected:" << expected.to_str();
}

TYPED_TEST(TileTest, ZeroSizeInput) {
  using T = TypeParam;

  auto values = column_wrapper<T>{};

  auto expected = column_wrapper<T>{};

  cudf::table result = cudf::tile({values.get()}, 0);
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str() << "Expected:" << expected.to_str();
}
