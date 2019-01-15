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


#include "copying.hpp"
#include "types.hpp"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <thrust/device_vector.h>

struct ScatterTest : GdfTest {

};


TEST_F(ScatterTest, FirstTest){

    gdf_size_type source_size{100};
    gdf_size_type destination_size{100};

    std::vector<int32_t> source_data(source_size);
    std::iota(source_data.begin(), source_data.end(), 0);
    auto source_column = init_gdf_column(source_data, 0, [](int row, int col){return true;});

    thrust::device_vector<gdf_index_type> scatter_map(source_data.size());
    std::iota(scatter_map.begin(), scatter_map.end(), 0);

    std::vector<int32_t> destination_data(destination_size);
    auto destination_column = init_gdf_column(destination_data, 0, [](int row, int col){return true;});

    gdf_column * raw_source = source_column.get();
    gdf_column * raw_destination = destination_column.get();

    cudf::table source_table{ &raw_source, 1};
    cudf::table destination_table{ &raw_destination, 1};

    cudf::scatter(&source_table, scatter_map.data().get(), &destination_table);

    EXPECT_TRUE(gdf_equal_columns<int32_t>(raw_source, raw_destination));

    EXPECT_EQ(true,true);
}