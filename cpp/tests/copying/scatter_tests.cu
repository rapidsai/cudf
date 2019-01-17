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
#include "copying.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "types.hpp"

struct ScatterTest : GdfTest {};

TEST_F(ScatterTest, FirstTest) {
  gdf_size_type source_size{100};
  gdf_size_type destination_size{100};

  cudf::test::column_wrapper<int32_t> source_column{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};

  thrust::device_vector<gdf_index_type> scatter_map(source_size);
  thrust::sequence(scatter_map.begin(), scatter_map.end());

  cudf::test::column_wrapper<int32_t> destination_column(destination_size);

  gdf_column * raw_source = source_column.get();
  gdf_column * raw_destination = destination_column.get();

  cudf::table source_table{ &raw_source, 1};
  cudf::table destination_table{ &raw_destination, 1};

  cudf::scatter(&source_table, scatter_map.data().get(), &destination_table);

  EXPECT_TRUE(source_column == destination_column);
}

