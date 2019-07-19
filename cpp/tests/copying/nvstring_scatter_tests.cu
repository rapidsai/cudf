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
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/table.hpp>
#include <random>
#include <tests/utilities/nvcategory_utils.cuh>
#include <tests/utilities/valid_vectors.h>

template <typename T>
struct ScatterTest : GdfTest {};

TEST(ScatterTest, ScatterNVString)
{
  bool print = true;
  const int rows_size = 8;
  const size_t length = 1;

  const char** left_string_data = cudf::test::generate_string_data(rows_size, length, print);
  const char** right_string_data = cudf::test::generate_string_data(rows_size*2, length, print);

  std::vector<std::string> left_host_column (left_string_data, left_string_data + rows_size);
  std::vector<std::string> right_host_column (right_string_data, right_string_data + rows_size*2);

  gdf_column* left_column = cudf::test::create_nv_category_column_strings(left_string_data, rows_size);
  gdf_column* right_column = cudf::test::create_nv_category_column_strings(right_string_data, rows_size*2);

  if(print){
    print_gdf_column(left_column);
    print_gdf_column(right_column);
  }  
  
  std::vector<gdf_index_type> scatter_map({0, 1, 2, 6, 8, 10, 12, 14});
  rmm::device_vector<gdf_index_type> d_scatter_map = scatter_map;

  cudf::table source_table({left_column});
  cudf::table destination_table({right_column});

  cudf::scatter(&source_table, d_scatter_map.data().get(), &destination_table);
  
  if(print){
    print_gdf_column(left_column);
    print_gdf_column(right_column);
  }
/*
  std::vector<std::string> strs;
  std::vector<gdf_valid_type> valids;
  std::tie(strs, valids) = cudf::test::nvcategory_column_to_host(right_column);
  for(auto str : strs){
    printf("%s\t", str.c_str());
  }
  printf("\n");
*/ 
}

