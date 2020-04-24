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
#include <thrust/device_vector.h>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/table.hpp>
#include <random>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/scalar_wrapper.cuh>

template <typename T>
struct ScalarScatterTest : GdfTest {
};

using test_types = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8>;
TYPED_TEST_CASE(ScalarScatterTest, test_types);

TYPED_TEST(ScalarScatterTest, DestMissingValid)
{
  constexpr cudf::size_type target_size{1920};

  static_assert(0 == target_size % 3, "Size of source data must be a multiple of 3.");

  // elements with indices = 1,2 mod 3 are null
  cudf::test::column_wrapper<TypeParam> target_column(
    target_size,
    [](cudf::size_type row) { return static_cast<TypeParam>(row); },
    [](cudf::size_type row) { return row % 3 != 0; });

  // Scatter null values to the last half of the target column
  std::vector<cudf::size_type> host_scatter_map(target_size / 3);
  for (cudf::size_type i = 0; i < target_size / 3; ++i) { host_scatter_map[i] = i * 3; }
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  gdf_column* raw_target = target_column.get();

  cudf::table target_table{&raw_target, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), true);
  std::vector<gdf_scalar> source_row{*source.get()};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table = cudf::scatter(
                    source_row, scatter_map.data().get(), target_size / 3, target_table));

  // Copy result of target column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  EXPECT_EQ(destination_column.get()->null_count, 0);

  for (cudf::size_type i = 0; i < target_size / 3; i++) {
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i * 3]);
  }

  destination_table.destroy();
}

TYPED_TEST(ScalarScatterTest, ScatterMultiColValid)
{
  constexpr cudf::size_type target_size{1920};
  constexpr cudf::size_type n_cols = 3;

  static_assert(0 == target_size % 3, "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the target column
  std::vector<cudf::size_type> host_scatter_map(target_size / 3);
  for (cudf::size_type i = 0; i < target_size / 3; ++i) { host_scatter_map[i] = i * 3; }
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  std::vector<cudf::test::column_wrapper<TypeParam>> v_dest(
    n_cols,
    {target_size,
     [](cudf::size_type row) { return static_cast<TypeParam>(row); },
     [](cudf::size_type row) { return false; }});
  std::vector<gdf_column*> vp_dest{n_cols};
  for (size_t i = 0; i < v_dest.size(); i++) { vp_dest[i] = v_dest[i].get(); }

  cudf::table target_table{vp_dest};

  cudf::test::scalar_wrapper<TypeParam> source0(static_cast<TypeParam>(0), true);
  cudf::test::scalar_wrapper<TypeParam> source1(static_cast<TypeParam>(1), true);
  cudf::test::scalar_wrapper<TypeParam> source2(static_cast<TypeParam>(2), true);

  std::vector<gdf_scalar> source_row{*source0.get(), *source1.get(), *source2.get()};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table = cudf::scatter(
                    source_row, scatter_map.data().get(), target_size / 3, target_table));

  for (int c = 0; c < n_cols; c++) {
    // Copy result of target column to host
    std::vector<TypeParam> result_data;
    std::vector<cudf::valid_type> result_bitmask;
    cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(c));
    std::tie(result_data, result_bitmask) = destination_column.to_host();

    EXPECT_EQ(destination_column.get()->null_count, target_size * 2 / 3);

    for (cudf::size_type i = 0; i < target_size / 3; i++) {
      EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i * 3))
        << "Value at index " << i << " should be non-null!\n";
      EXPECT_EQ(static_cast<TypeParam>(c), result_data[i * 3]);
    }
  }

  destination_table.destroy();
}

TYPED_TEST(ScalarScatterTest, DISABLED_ScatterValid)
{
  constexpr cudf::size_type target_size{1920};

  static_assert(0 == target_size % 3, "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the target column
  std::vector<cudf::size_type> host_scatter_map(target_size / 3);
  for (cudf::size_type i = 0; i < target_size / 3; ++i) { host_scatter_map[i] = i * 3; }
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> target_column(target_size, true);

  gdf_column* raw_target = target_column.get();

  cudf::table target_table{&raw_target, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), true);

  std::vector<gdf_scalar> source_row{*source.get()};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table = cudf::scatter(
                    source_row, scatter_map.data().get(), target_size / 3, target_table));

  // Copy result of target column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  EXPECT_EQ(destination_column.get()->null_count, target_size * 2 / 3);

  for (cudf::size_type i = 0; i < target_size / 3; i++) {
    EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i * 3))
      << "Value at index " << i << " should be non-null!\n";
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i * 3]);
  }

  destination_table.destroy();
}

TYPED_TEST(ScalarScatterTest, DISABLED_ScatterNull)
{
  constexpr cudf::size_type target_size{1920};

  static_assert(0 == target_size % 3, "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the target column
  std::vector<cudf::size_type> host_scatter_map(target_size / 3);
  for (cudf::size_type i = 0; i < target_size / 3; ++i) { host_scatter_map[i] = i * 3 + 1; }
  thrust::device_vector<cudf::size_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> target_column(target_size, false);

  gdf_column* raw_target = target_column.get();

  cudf::table target_table{&raw_target, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), false);  // valid = false

  std::vector<gdf_scalar> source_row{*source.get()};

  cudf::table destination_table;

  EXPECT_NO_THROW(destination_table = cudf::scatter(
                    source_row, scatter_map.data().get(), target_size / 3, target_table));

  // Copy result of target column to host
  std::vector<TypeParam> result_data;
  std::vector<cudf::valid_type> result_bitmask;
  cudf::test::column_wrapper<TypeParam> destination_column(*destination_table.get_column(0));
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  EXPECT_EQ(destination_column.get()->null_count, target_size / 3);

  for (cudf::size_type i = 0; i < target_size / 3; i++) {
    EXPECT_FALSE(gdf_is_valid(result_bitmask.data(), i * 3 + 1))
      << "Value at index " << i << " should be null!\n";
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i * 3 + 1]);
  }

  destination_table.destroy();
}
