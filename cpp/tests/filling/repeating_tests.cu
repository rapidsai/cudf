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

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

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

TYPED_TEST(RepeatTest, SetRanges)
{
  using T = TypeParam;

  // Making the ranges that will be filled
  gdf_size_type num_ranges = 10;
  gdf_size_type max_range_size = 10;

  std::vector<gdf_size_type> range_sizes(num_ranges);
  std::transform(range_sizes.begin(), range_sizes.end(), range_sizes.begin(),
    [&](gdf_size_type i) { return random_int(0, max_range_size); });

  std::vector<int> values_data(num_ranges);
  std::iota(values_data.begin(), values_data.end(), 0);
  std::transform(values_data.begin(), values_data.end(), values_data.begin(),
    [](gdf_size_type i) { return i * 2; });

  // set your expectations
  gdf_size_type column_size = std::accumulate(range_sizes.begin(), range_sizes.end(), 0);
  std::vector<int> expect_vals;
  for (size_t i = 0; i < range_sizes.size(); i++) {
    for (gdf_size_type j = 0; j < range_sizes[i]; j++) {
      expect_vals.push_back(values_data[i]);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(num_ranges,
    [&](gdf_size_type i) { return values_data[i]; });

  column_wrapper<gdf_size_type> counts(range_sizes);

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; });

  column_wrapper<T> actual(cudf::repeat(*values.get(), *counts.get()));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}

TYPED_TEST(RepeatTest, SetRangesNullable)
{
  using T = TypeParam;

  // Making the ranges that will be filled
  gdf_size_type num_ranges = 10;
  gdf_size_type max_range_size = 10;

  std::vector<gdf_size_type> range_sizes(num_ranges);
  std::transform(range_sizes.begin(), range_sizes.end(), range_sizes.begin(),
    [&](gdf_size_type i) { return random_int(0, max_range_size); });
  
  std::vector<gdf_size_type> range_offsets(num_ranges);
  std::partial_sum(range_sizes.begin(), range_sizes.end(), range_offsets.begin());

  std::vector<int> values_data(num_ranges);
  std::iota(values_data.begin(), values_data.end(), 0);
  std::transform(values_data.begin(), values_data.end(), values_data.begin(),
    [](gdf_size_type i) { return i * 2; });

  // set your expectations
  gdf_size_type column_size = std::accumulate(range_sizes.begin(), range_sizes.end(), 0);
  std::vector<int> expect_vals;
  for (size_t i = 0; i < range_sizes.size(); i++) {
    for (gdf_size_type j = 0; j < range_sizes[i]; j++) {
      expect_vals.push_back(values_data[i]);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> values = factory.make(num_ranges,
    [&](gdf_size_type i) { return values_data[i]; },
    [&](gdf_size_type i) { return i % 2; });

  column_wrapper<gdf_size_type> counts(range_sizes);

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; },
    [&](gdf_index_type row) { 
      auto corresponding_value_it = std::upper_bound(range_offsets.begin(), range_offsets.end(), row);
      return (corresponding_value_it - range_offsets.begin()) % 2; });

  column_wrapper<T> actual(cudf::repeat(*values.get(), *counts.get()));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}

TYPED_TEST(RepeatTest, ZeroSizeInput)
{
  using T = TypeParam;

  auto values = column_wrapper<T>{};
  auto counts = column_wrapper<gdf_size_type>{};

  auto expected = column_wrapper<T>{};

  column_wrapper<T> actual(cudf::repeat(*values.get(), *counts.get()));

  EXPECT_EQ(expected, actual) << "  Actual:" << actual.to_str()
                              << "Expected:" << expected.to_str();
}
