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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/valid_vectors.h>
#include <thrust/device_vector.h>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/table.hpp>
#include <random>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/nvcategory_utils.cuh>

template <typename T>
struct ScatterTest : GdfTest {
};

using test_types = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8>;
TYPED_TEST_CASE(ScatterTest, test_types);

TYPED_TEST(ScatterTest, DtypeMistach)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type target_size{1000};

  cudf::test::column_wrapper<int32_t> source(source_size);
  cudf::test::column_wrapper<float> destination(target_size);

  gdf_column* raw_source      = source.get();
  gdf_column* raw_destination = destination.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table target_table{&raw_destination, 1};

  rmm::device_vector<cudf::size_type> scatter_map(source_size);

  cudf::table destination_table;

  EXPECT_THROW(
    destination_table = cudf::scatter(source_table, scatter_map.data().get(), target_table),
    cudf::logic_error);

  destination_table.destroy();
}

TYPED_TEST(ScatterTest, NumColumnsMismatch)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type target_size{1000};

  cudf::test::column_wrapper<TypeParam> source0(source_size, true);
  cudf::test::column_wrapper<TypeParam> source1(source_size, true);
  cudf::test::column_wrapper<TypeParam> destination(target_size, false);

  std::vector<gdf_column*> source_cols{source0.get(), source1.get()};

  gdf_column* raw_destination = destination.get();

  cudf::table source_table{source_cols.data(), 2};
  cudf::table target_table{&raw_destination, 1};

  rmm::device_vector<cudf::size_type> scatter_map(source_size);

  cudf::table destination_table;

  EXPECT_THROW(
    destination_table = cudf::scatter(source_table, scatter_map.data().get(), target_table),
    cudf::logic_error);

  destination_table.destroy();
}

// This also test the case where the source column has a valid bitmask while
// the target column does not.
TYPED_TEST(ScatterTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type target_size{1000};

  cudf::test::column_wrapper<TypeParam> source_column(
    source_size,
    [](cudf::size_type row) { return static_cast<TypeParam>(row); },
    [](cudf::size_type row) { return row != source_size / 2; });

  thrust::device_vector<cudf::size_type> scatter_map(source_size);
  thrust::sequence(scatter_map.begin(), scatter_map.end());

  cudf::test::column_wrapper<TypeParam> target_column(target_size, false);
  gdf_column* raw_source      = source_column.get();
  gdf_column* raw_destination = target_column.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table target_table{&raw_destination, 1};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table =
                    cudf::scatter(source_table, scatter_map.data().get(), target_table));

  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));

  EXPECT_TRUE(source_column == destination_column);

  destination_table.destroy();
}

TYPED_TEST(ScatterTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type target_size{1000};

  cudf::test::column_wrapper<TypeParam> source_column(
    source_size,
    [](cudf::size_type row) { return static_cast<TypeParam>(row); },
    [](cudf::size_type row) { return true; });

  // Create scatter_map that reverses order of source_column
  std::vector<cudf::size_type> host_scatter_map(source_size);
  std::iota(host_scatter_map.begin(), host_scatter_map.end(), 0);
  std::reverse(host_scatter_map.begin(), host_scatter_map.end());
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> target_column(target_size, true);

  gdf_column* raw_source      = source_column.get();
  gdf_column* raw_destination = target_column.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table target_table{&raw_destination, 1};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table =
                    cudf::scatter(source_table, scatter_map.data().get(), target_table));

  // Expected result is the reversal of the source column
  std::vector<TypeParam> expected_data;
  std::vector<cudf::valid_type> expected_bitmask;
  std::tie(expected_data, expected_bitmask) = source_column.to_host();
  std::reverse(expected_data.begin(), expected_data.end());

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  for (cudf::size_type i = 0; i < target_size; i++) {
    EXPECT_EQ(expected_data[i], result_data[i]) << "Data at index " << i << " doesn't match!\n";
    EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i))
      << "Value at index " << i << " should be non-null!\n";
  }

  destination_table.destroy();
}

TYPED_TEST(ScatterTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};
  constexpr cudf::size_type target_size{1000};

  // source column has all null values
  cudf::test::column_wrapper<TypeParam> source_column(
    source_size,
    [](cudf::size_type row) { return static_cast<TypeParam>(row); },
    [](cudf::size_type row) { return false; });

  // Create scatter_map that scatters to random locations
  std::vector<cudf::size_type> host_scatter_map(source_size);
  std::iota(host_scatter_map.begin(), host_scatter_map.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_scatter_map.begin(), host_scatter_map.end(), g);
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> target_column(target_size, true);

  gdf_column* raw_source      = source_column.get();
  gdf_column* raw_destination = target_column.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table target_table{&raw_destination, 1};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table =
                    cudf::scatter(source_table, scatter_map.data().get(), target_table));

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  // All values of result should be null
  for (cudf::size_type i = 0; i < target_size; i++) {
    EXPECT_FALSE(gdf_is_valid(result_bitmask.data(), i))
      << "Value at index " << i << " should be null!\n";
  }

  destination_table.destroy();
}

TYPED_TEST(ScatterTest, EveryOtherNull)
{
  constexpr cudf::size_type source_size{1234};
  constexpr cudf::size_type target_size{source_size};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");
  static_assert(source_size == target_size, "Source and destination columns must be equal size.");

  // elements with even indices are null
  cudf::test::column_wrapper<TypeParam> source_column(
    source_size,
    [](cudf::size_type row) { return static_cast<TypeParam>(row); },
    [](cudf::size_type row) { return row % 2; });

  // Scatter null values to the last half of the destination column
  std::vector<cudf::size_type> host_scatter_map(source_size);
  for (cudf::size_type i = 0; i < source_size / 2; ++i) {
    host_scatter_map[i * 2]     = target_size / 2 + i;
    host_scatter_map[i * 2 + 1] = i;
  }
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> target_column(target_size, true);

  gdf_column* raw_source      = source_column.get();
  gdf_column* raw_destination = target_column.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table target_table{&raw_destination, 1};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table =
                    cudf::scatter(source_table, scatter_map.data().get(), target_table));

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  for (cudf::size_type i = 0; i < target_size; i++) {
    // The first half of the destination column should be all valid
    // and values should be 1, 3, 5, 7, etc.
    if (i < target_size / 2) {
      EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i))
        << "Value at index " << i << " should be non-null!\n";
      EXPECT_EQ(static_cast<TypeParam>(1 + i * 2), result_data[i]);
    }
    // The last half of the destination column should be all null
    else {
      EXPECT_FALSE(gdf_is_valid(result_bitmask.data(), i))
        << "Value at index " << i << " should be null!\n";
    }
  }

  destination_table.destroy();
}

// The test to test against BUG #2007
TYPED_TEST(ScatterTest, PreserveDestBitmask)
{
  cudf::test::column_wrapper<int64_t> source_column({10, -1}, [](auto index) { return false; });
  // So source is {@, @}
  cudf::test::column_wrapper<int64_t> target_column({10, -1, 6, 7},
                                                    [](auto index) { return index != 2; });
  // So destination is {10, -1, @, 7}

  std::vector<cudf::size_type> scatter_map({1, 3});
  rmm::device_vector<cudf::size_type> d_scatter_map = scatter_map;

  cudf::table source_table({source_column.get()});
  cudf::table target_table({target_column.get()});

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table =
                    cudf::scatter(source_table, d_scatter_map.data().get(), target_table));

  cudf::test::column_wrapper<int64_t> destination_column(*destination_table.get_column(0));

  // We should expect {10, @, @, @}
  cudf::test::column_wrapper<int64_t> expect({10, 10, 6, -1},
                                             [](auto index) { return index == 0; });
  EXPECT_EQ(expect, destination_column);

  destination_table.destroy();
}
