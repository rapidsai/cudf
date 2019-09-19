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

#include <cudf/filling.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/column_wrapper_factory.hpp>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <numeric>
#include <random>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
using scalar_wrapper = cudf::test::scalar_wrapper<T>;

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

template <typename T>
struct RepeatTest : GdfTest {};

using test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(RepeatTest, test_types);

TYPED_TEST(RepeatTest, RepeatScalarCount)
{
  using T = TypeParam;

  gdf_size_type repeat_count = 10;
  gdf_size_type num_values = 10;

  // set your expectations
  gdf_size_type column_size = num_values * repeat_count;
  std::vector<int> expect_vals;
  for (gdf_size_type i = 0; i < num_values; i++) {
    for (gdf_size_type j = 0; j < repeat_count; j++) {
      expect_vals.push_back(i);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(num_values,
    [&](gdf_size_type i) { return i; });

  scalar_wrapper<gdf_size_type> count{repeat_count};

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; });

  cudf::table result = cudf::repeat({values.get()}, *count.get());
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}

TYPED_TEST(RepeatTest, Repeat)
{
  using T = TypeParam;

  // Making the ranges that will be filled
  gdf_size_type num_counts = 10;
  gdf_size_type max_count = 10;

  std::vector<gdf_size_type> counts_data(num_counts);
  std::transform(counts_data.begin(), counts_data.end(), counts_data.begin(),
    [&](gdf_size_type i) { return random_int(0, max_count); });

  std::vector<int> values_data(num_counts);
  std::iota(values_data.begin(), values_data.end(), 0);
  std::transform(values_data.begin(), values_data.end(), values_data.begin(),
    [](gdf_size_type i) { return i * 2; });

  // set your expectations
  gdf_size_type column_size = std::accumulate(counts_data.begin(), counts_data.end(), 0);
  std::vector<int> expect_vals;
  for (size_t i = 0; i < counts_data.size(); i++) {
    for (gdf_size_type j = 0; j < counts_data[i]; j++) {
      expect_vals.push_back(values_data[i]);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(num_counts,
    [&](gdf_size_type i) { return values_data[i]; });

  column_wrapper<gdf_size_type> counts(counts_data);

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; });

  cudf::table result = cudf::repeat({values.get()}, *counts.get());
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}

TYPED_TEST(RepeatTest, RepeatNullable)
{
  using T = TypeParam;

  // Making the ranges that will be filled
  gdf_size_type num_counts = 10;
  gdf_size_type max_count = 10;

  std::vector<gdf_size_type> counts_data(num_counts);
  std::transform(counts_data.begin(), counts_data.end(), counts_data.begin(),
    [&](gdf_size_type i) { return random_int(0, max_count); });
  
  std::vector<gdf_size_type> offsets(num_counts);
  std::partial_sum(counts_data.begin(), counts_data.end(), offsets.begin());

  std::vector<int> values_data(num_counts);
  std::iota(values_data.begin(), values_data.end(), 0);
  std::transform(values_data.begin(), values_data.end(), values_data.begin(),
    [](gdf_size_type i) { return i * 2; });

  // set your expectations
  gdf_size_type column_size = std::accumulate(counts_data.begin(), counts_data.end(), 0);
  std::vector<int> expect_vals;
  for (size_t i = 0; i < counts_data.size(); i++) {
    for (gdf_size_type j = 0; j < counts_data[i]; j++) {
      expect_vals.push_back(values_data[i]);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(num_counts,
    [&](gdf_size_type i) { return values_data[i]; },
    [&](gdf_size_type i) { return i % 2; });

  column_wrapper<gdf_size_type> counts(counts_data);

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; },
    [&](gdf_index_type row) { 
      auto corresponding_value_it = std::upper_bound(offsets.begin(), offsets.end(), row);
      return (corresponding_value_it - offsets.begin()) % 2; });

  cudf::table result = cudf::repeat({values.get()}, *counts.get());
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}

TYPED_TEST(RepeatTest, ZeroSizeInput)
{
  using T = TypeParam;

  auto values = column_wrapper<T>{};
  auto counts = column_wrapper<gdf_size_type>{};

  auto expected = column_wrapper<T>{};

  cudf::table result = cudf::repeat({values.get()}, *counts.get());
  column_wrapper<T> actual(*result.get_column(0));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}
