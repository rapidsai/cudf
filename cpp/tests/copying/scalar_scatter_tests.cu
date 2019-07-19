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

#include <thrust/device_vector.h>
#include <cudf/copying.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/table.hpp>
#include <random>

template <typename T>
struct ScalarScatterTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8>;
TYPED_TEST_CASE(ScalarScatterTest, test_types);

TYPED_TEST(ScalarScatterTest, DestMissingValid) {
  constexpr gdf_size_type destination_size{1920};

  static_assert(0 == destination_size % 3,
                "Size of source data must be a multiple of 3.");

  // elements with even indices are null
  cudf::test::column_wrapper<TypeParam> source_column(
      destination_size,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return row % 3 != 0; });

  // Scatter null values to the last half of the destination column
  std::vector<gdf_index_type> host_scatter_map(destination_size/3);
  for (gdf_size_type i = 0; i < destination_size/3; ++i) {
    host_scatter_map[i] = i*3;
  }
  thrust::device_vector<gdf_index_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> destination_column(destination_size,
                                                           false);

  gdf_column* raw_destination = destination_column.get();

  cudf::table destination_table{&raw_destination, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), true);
  std::vector<gdf_scalar*> source_row {source.get()};
  
  cudf::test::scalar_wrapper<TypeParam> null_source(static_cast<TypeParam>(1), false);
  std::vector<gdf_scalar*> null_source_row {null_source.get()};
  
  EXPECT_THROW(cudf::scatter(null_source_row, scatter_map.data().get(), destination_size/3,
                                &destination_table), cudf::logic_error);

  EXPECT_NO_THROW(cudf::scatter(source_row, scatter_map.data().get(), destination_size/3,
                                &destination_table));

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<gdf_valid_type> result_bitmask;
  std::tie(result_data, result_bitmask) = destination_column.to_host();
  
  EXPECT_EQ(raw_destination->null_count, 0);

  for (gdf_index_type i = 0; i < destination_size/3; i++) {
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i*3]);
  }
}

TYPED_TEST(ScalarScatterTest, ScatterMultiColValid) {
  constexpr gdf_size_type destination_size{1920};
  constexpr gdf_size_type n_cols = 3;

  static_assert(0 == destination_size % 3,
                "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the destination column
  std::vector<gdf_index_type> host_scatter_map(destination_size/3);
  for (gdf_size_type i = 0; i < destination_size/3; ++i) {
    host_scatter_map[i] = i*3;
  }
  thrust::device_vector<gdf_index_type> scatter_map(host_scatter_map);
  
  std::vector<cudf::test::column_wrapper<TypeParam>> v_dest(
    n_cols,
    { destination_size, 
      [](gdf_index_type row){return static_cast<TypeParam>(row);},
      [](gdf_index_type row) { return false; }
    }
  );
  std::vector<gdf_column*> vp_dest {n_cols};
  for(size_t i = 0; i < v_dest.size(); i++){
    vp_dest[i] = v_dest[i].get();  
  }
 
  cudf::table destination_table{ vp_dest };

  cudf::test::scalar_wrapper<TypeParam> source0(static_cast<TypeParam>(0), true);
  cudf::test::scalar_wrapper<TypeParam> source1(static_cast<TypeParam>(1), true);
  cudf::test::scalar_wrapper<TypeParam> source2(static_cast<TypeParam>(2), true);
 
  std::vector<gdf_scalar*> source_row {source0.get(), source1.get(), source2.get()};

  EXPECT_NO_THROW(cudf::scatter(source_row, scatter_map.data().get(), destination_size/3,
                                &destination_table));

  for(int c = 0; c < n_cols; c++){
    // Copy result of destination column to host
    std::vector<TypeParam> result_data;
    std::vector<gdf_valid_type> result_bitmask;
    std::tie(result_data, result_bitmask) = v_dest[c].to_host();
    
    EXPECT_EQ(vp_dest[c]->null_count, destination_size*2/3);

    for (gdf_index_type i = 0; i < destination_size/3; i++) {
      EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i*3))
          << "Value at index " << i << " should be non-null!\n";
      EXPECT_EQ(static_cast<TypeParam>(c), result_data[i*3]);
    }
  }
}

TYPED_TEST(ScalarScatterTest, ScatterValid) {
  constexpr gdf_size_type destination_size{1920};

  static_assert(0 == destination_size % 3,
                "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the destination column
  std::vector<gdf_index_type> host_scatter_map(destination_size/3);
  for (gdf_size_type i = 0; i < destination_size/3; ++i) {
    host_scatter_map[i] = i*3;
  }
  thrust::device_vector<gdf_index_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> destination_column(destination_size,
                                                           true);

  gdf_column* raw_destination = destination_column.get();

  cudf::table destination_table{&raw_destination, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), true);
 
  std::vector<gdf_scalar*> source_row {source.get()};

  EXPECT_NO_THROW(cudf::scatter(source_row, scatter_map.data().get(), destination_size/3,
                                &destination_table));

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<gdf_valid_type> result_bitmask;
  std::tie(result_data, result_bitmask) = destination_column.to_host();
  
  EXPECT_EQ(raw_destination->null_count, destination_size*2/3);

  for (gdf_index_type i = 0; i < destination_size/3; i++) {
    EXPECT_TRUE(gdf_is_valid(result_bitmask.data(), i*3))
        << "Value at index " << i << " should be non-null!\n";
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i*3]);
  }
}
TYPED_TEST(ScalarScatterTest, ScatterNull) {
  constexpr gdf_size_type destination_size{1920};

  static_assert(0 == destination_size % 3,
                "Size of source data must be a multiple of 3.");

  // Scatter null values to the last half of the destination column
  std::vector<gdf_index_type> host_scatter_map(destination_size/3);
  for (gdf_size_type i = 0; i < destination_size/3; ++i) {
    host_scatter_map[i] = i*3+1;
  }
  thrust::device_vector<gdf_index_type> scatter_map(host_scatter_map);

  cudf::test::column_wrapper<TypeParam> destination_column(destination_size,
                                                           true);

  gdf_column* raw_destination = destination_column.get();

  cudf::table destination_table{&raw_destination, 1};

  cudf::test::scalar_wrapper<TypeParam> source(static_cast<TypeParam>(1), false); // valid = false
 
  std::vector<gdf_scalar*> source_row {source.get()};

  EXPECT_NO_THROW(cudf::scatter(source_row, scatter_map.data().get(), destination_size/3,
                                &destination_table));

  // Copy result of destination column to host
  std::vector<TypeParam> result_data;
  std::vector<gdf_valid_type> result_bitmask;
  std::tie(result_data, result_bitmask) = destination_column.to_host();

  EXPECT_EQ(raw_destination->null_count, destination_size);
  
  for (gdf_index_type i = 0; i < destination_size/3; i++) {
    EXPECT_FALSE(gdf_is_valid(result_bitmask.data(), i*3+1))
        << "Value at index " << i << " should be null!\n";
    EXPECT_EQ(static_cast<TypeParam>(1), result_data[i*3+1]);
  }
}

