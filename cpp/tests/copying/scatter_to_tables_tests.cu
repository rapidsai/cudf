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

#include<cudf/copying.hpp>
#include<cudf/legacy/table.hpp>

#include<gtest/gtest.h>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include<tests/copying/copying_test_helper.hpp>
#include <bitmask/legacy/bit_mask.cuh>
#include<random>

template<typename T>
struct ScatterToTablesTest : GdfTest {};

using test_types = 
::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8>;
TYPED_TEST_CASE(ScatterToTablesTest, test_types);

//  All Pass test cases 
TYPED_TEST(ScatterToTablesTest, PassTest) {
  constexpr gdf_size_type table_n_cols{2};
  constexpr gdf_size_type table_n_rows{1000};
  constexpr gdf_size_type scatter_size{table_n_rows};

  std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(table_n_cols, {table_n_rows, 
  [](gdf_index_type row) {return static_cast<TypeParam>(row);},
  [](gdf_index_type row) { return false;}
  });
  
  cudf::test::column_wrapper<gdf_index_type> scatter_map(scatter_size, 
  [](gdf_index_type row) {return static_cast<gdf_index_type>(row%2);},
  false
  );
  
  std::vector<gdf_column*> v_cols(table_n_cols);
  for(size_t i=0; i<v_colwrap.size(); i++)
    v_cols[i]=v_colwrap[i].get();
  
  cudf::table input_table{v_cols};
  std::vector<cudf::table> output_tables;
  EXPECT_NO_THROW(output_tables =
                  cudf::scatter_to_tables(input_table, scatter_map));
  EXPECT_EQ(2U, output_tables.size());
  for(auto table : output_tables) {
    EXPECT_EQ( table_n_cols, table.num_columns());
    EXPECT_EQ( table_n_rows/2, table.num_rows());
  }
}

// Test Failure if scatter_map is not `GDF_INT32` column
TYPED_TEST(ScatterToTablesTest, ScatterTypeFailTest) {
  constexpr gdf_size_type table_n_cols{2};
  constexpr gdf_size_type table_n_rows{1000};
  constexpr gdf_size_type scatter_size{table_n_rows};

  //data
  std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(
    table_n_cols, {table_n_rows,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return false; }});

  cudf::test::column_wrapper<TypeParam> scatter_map(scatter_size, 
  [](gdf_index_type row) {return static_cast<TypeParam>(row%2);}, false); 

  std::vector<gdf_column*> v_cols(table_n_cols);
  for(size_t i=0; i<v_colwrap.size(); i++)
    v_cols[i]=v_colwrap[i].get();
  
  //input datatypes
  cudf::table input_table{v_cols};
  //test
  if (std::is_same<gdf_index_type, TypeParam>())
    EXPECT_NO_THROW(cudf::scatter_to_tables(input_table, scatter_map));
  else
    CUDF_EXPECT_THROW_MESSAGE(cudf::scatter_to_tables(input_table, scatter_map),
    "scatter_map is not GDF_INT32 column.");
}

// Test Failure if scatter_map column has null entries
TYPED_TEST(ScatterToTablesTest, NullMapTest) {
  constexpr gdf_size_type table_n_cols{2};
  constexpr gdf_size_type table_n_rows{1000};
  constexpr gdf_size_type scatter_size{table_n_rows};

  //data
  std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(
    table_n_cols, {table_n_rows,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return false; }});

  std::vector<gdf_column*> v_cols(table_n_cols);
  for(size_t i=0; i<v_colwrap.size(); i++)
    v_cols[i]=v_colwrap[i].get();
  
  //input datatypes
  cudf::table input_table{v_cols};
 
  //1. No-Null
  cudf::test::column_wrapper<gdf_index_type> scatter_map1(scatter_size, 
  [](gdf_index_type row) {return row%2;}, 
  [](gdf_index_type row) { return true;}); 
 EXPECT_NO_THROW(cudf::scatter_to_tables(input_table, scatter_map1));
  //2. All-Null
  cudf::test::column_wrapper<gdf_index_type> scatter_map2(scatter_size, 
  [](gdf_index_type row) {return row%2;}, 
  [](gdf_index_type row) { return false;}); 
 CUDF_EXPECT_THROW_MESSAGE(cudf::scatter_to_tables(input_table, scatter_map2),
                           "Scatter map cannot contain null elements.");
  //3. Some-Null
  cudf::test::column_wrapper<gdf_index_type> scatter_map3(scatter_size, 
  [](gdf_index_type row) {return row%2;}, 
  [](gdf_index_type row) { return rand()%3==0;}); 
 CUDF_EXPECT_THROW_MESSAGE(cudf::scatter_to_tables(input_table, scatter_map3),
                           "Scatter map cannot contain null elements.");
 }

// Test Failure if scatter_map and input_table number of rows mismatch are not equal
TYPED_TEST(ScatterToTablesTest, SizeMismatchTest) {
  constexpr gdf_size_type table_n_cols{2};
  constexpr gdf_size_type table_n_rows{1000};

  //data
  std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(
    table_n_cols, {table_n_rows,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return false; }});

  std::vector<gdf_column*> v_cols(table_n_cols);
  for(size_t i=0; i<v_colwrap.size(); i++)
    v_cols[i]=v_colwrap[i].get();
  
  //input datatypes
  cudf::table input_table{v_cols};
 
  //1. Equal
  cudf::test::column_wrapper<gdf_index_type> scatter_map1(table_n_rows, 
      [](gdf_index_type row) { return row; }, false);
 EXPECT_NO_THROW(cudf::scatter_to_tables(input_table, scatter_map1));
  //2. Lesser
  cudf::test::column_wrapper<gdf_index_type> scatter_map2(table_n_rows/2, 
      [](gdf_index_type row) { return row; }, false);
  CUDF_EXPECT_THROW_MESSAGE(cudf::scatter_to_tables(input_table, scatter_map2),
    "scatter_map length is not equal to number of rows in input table.");
  //3. Greater
  cudf::test::column_wrapper<gdf_index_type> scatter_map3(table_n_rows * 2,
      [](gdf_index_type row) { return row; }, false);
  CUDF_EXPECT_THROW_MESSAGE(cudf::scatter_to_tables(input_table, scatter_map3),
    "scatter_map length is not equal to number of rows in input table.");
 }

// Test expected output if some of inputs sizes are zero.
/*
* c   = number of input table columns
* r   = number of input table rows
* sm  = number of rows in scatter_map column
* o/p = number of output tables
*
* c r sm  RESULT
* 0 * *   seg fault (zero columns is not allowed)
* 1 0 0   pass  (o/p=0)
* 1 0 1   throw (r!=sm)
* 1 1 0   throw (r!=sm)
* 1 1 1   pass  (o/p=1)
*
*/
TYPED_TEST(ScatterToTablesTest, ZeroSizeTest) {
  gdf_size_type table_n_cols=1;
  for(gdf_size_type table_n_rows : {0, 1}) {
    for(gdf_size_type scatter_size : {0, 1}) {
      //std::cout << table_n_cols << table_n_rows << scatter_size << std::endl;

      //data
      std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(
          table_n_cols, {table_n_rows,
          [](gdf_index_type row) { return static_cast<TypeParam>(row); },
          [](gdf_index_type row) { return false; }});
      std::vector<gdf_column *> v_cols(table_n_cols);
      for (size_t i = 0; i < v_colwrap.size(); i++)
        v_cols[i] = v_colwrap[i].get();

      //input datatypes
      cudf::table input_table{v_cols};
      cudf::test::column_wrapper<gdf_index_type> scatter_map(scatter_size,
          [](gdf_index_type row) { return row; }, false);
      std::vector<cudf::table> output_tables;
      if (table_n_cols == 0) table_n_rows=0;
      if(scatter_size != table_n_rows) {
        CUDF_EXPECT_THROW_MESSAGE(
            output_tables = cudf::scatter_to_tables(input_table, scatter_map),
            "scatter_map length is not equal to number of rows in input table.");
      } else {
        EXPECT_NO_THROW(
            output_tables = cudf::scatter_to_tables(input_table, scatter_map));
        EXPECT_EQ(size_t(scatter_size), output_tables.size());
      }
    }
  }
}

template <typename T>
auto scatter_columns(
    std::vector<std::vector<T>> input_cols_data,
    std::vector<std::vector<gdf_valid_type>> input_cols_bitmask,
    std::vector<gdf_index_type> scatter_map)
{
  auto max = *std::max_element(scatter_map.begin(), scatter_map.end());
  std::vector<std::vector<std::vector<T>>> output_cols_data(max+1);
  std::vector<std::vector<std::vector<gdf_valid_type>>> output_cols_bitmask(max+1);
  std::vector<std::vector<gdf_size_type>> output_cols_null_count(max+1);
  auto num_cols = input_cols_data.size();
  for (auto g = 0; g <= max; g++)
  {
    auto n_output = std::count(scatter_map.begin(), scatter_map.end(), g);
    for (size_t c = 0; c < num_cols; c++)
    {
      std::vector<T> output_col_data(n_output);
      std::vector<gdf_valid_type> output_col_bitmask(gdf_valid_allocation_size(n_output), ~gdf_valid_type(0));
      gdf_index_type j = 0, nullc=0;
      for (size_t r=0; r< scatter_map.size(); r++)
      {
        if (g == scatter_map[r])
        {
          output_col_data[j]=input_cols_data[c][r];
          if(!gdf_is_valid(input_cols_bitmask[c].data(), r))
          {
            bit_mask::clear_bit_unsafe(
              reinterpret_cast<bit_mask::bit_mask_t*>(output_col_bitmask.data()), j);
            nullc++;
          }
          j++;
        }
      }
      output_cols_data[g].push_back(output_col_data);
      output_cols_bitmask[g].push_back(output_col_bitmask);
      output_cols_null_count[g].push_back(nullc);
      EXPECT_EQ(n_output, j) << "Reference solution calculation failure\n";
    }
    EXPECT_EQ(num_cols, output_cols_data[g].size()) << "Reference solution calculation failure\n";
    EXPECT_EQ(num_cols, output_cols_bitmask[g].size()) << "Reference solution calculation failure\n";
    EXPECT_EQ(num_cols, output_cols_null_count[g].size()) << "Reference solution calculation failure\n";
  }
  EXPECT_EQ(size_t(max+1), output_cols_data.size()) << "Reference solution calculation failure\n";
  EXPECT_EQ(size_t(max+1), output_cols_bitmask.size()) << "Reference solution calculation failure\n";
  EXPECT_EQ(size_t(max+1), output_cols_null_count.size()) << "Reference solution calculation failure\n";
  return std::make_tuple(output_cols_data,
                         output_cols_bitmask,
                         output_cols_null_count);
}

TYPED_TEST(ScatterToTablesTest, FunctionalTest) {
  using TypeParam = gdf_index_type;
  constexpr gdf_size_type table_n_cols{2};
  constexpr gdf_size_type table_n_rows{1000};
  constexpr size_t scatter_size{table_n_rows};
  // table data
  std::vector<cudf::test::column_wrapper<TypeParam>> v_colwrap(
      table_n_cols, {table_n_rows,
                     [](gdf_index_type row) { return static_cast<TypeParam>(row); },
                     [](gdf_index_type row) { return rand()%3==0;}}); 
  std::vector<gdf_column *> v_cols(table_n_cols);
  for (size_t i = 0; i < v_colwrap.size(); i++)
    v_cols[i] = v_colwrap[i].get();
  cudf::table input_table{v_cols};

  // Transfer input columns to host
  std::vector<std::vector<TypeParam>> input_cols_data(table_n_cols);
  std::vector<std::vector<gdf_valid_type>> input_cols_bitmask(table_n_cols);
  for(int i=0; i < table_n_cols ; i++) {
    std::tie(input_cols_data[i], input_cols_bitmask[i]) = v_colwrap[i].to_host();
  }
  std::vector<std::vector<std::vector<TypeParam>>> output_cols_data;
  std::vector<std::vector<std::vector<gdf_valid_type>>> output_cols_bitmask;
  std::vector<std::vector<gdf_size_type>> output_cols_null_count;
  // run for all different ranges of value of scatter_map
  for (auto scatter_case : std::vector<std::pair<size_t, std::function<gdf_index_type(gdf_index_type)>>>{
           // range 0-0
           {1U, [](gdf_index_type row) { return 0; }},
           // range 0-1
           {2U, [](gdf_index_type row) { return row % 2; }},
           // range 0-(r/2-1)
           {scatter_size / 2, [](gdf_index_type row) { return row / 2; }},
           // range 0-(r-1)
           {scatter_size, [](gdf_index_type row) { return row; }},
           // range 0-(2*(r-1))
           {2 * scatter_size - 1, [](gdf_index_type row) { return 2 * row; }}})
  {
    cudf::test::column_wrapper<gdf_index_type> scatter_map(
        scatter_size,
        scatter_case.second, false);
    std::vector<cudf::table> output_tables;
    EXPECT_NO_THROW(
        output_tables = cudf::scatter_to_tables(input_table, scatter_map));
    EXPECT_EQ(scatter_case.first, output_tables.size());
    // Perform scatter in cpu
    std::tie(output_cols_data, output_cols_bitmask, output_cols_null_count) =
        scatter_columns(
            input_cols_data,
            input_cols_bitmask,
            std::get<0>(scatter_map.to_host()));
    //Compare GPU and CPU results
    EXPECT_EQ(output_cols_data.size(), output_tables.size()) << "Expected=" << scatter_case.first << std::endl;
    for (size_t i = 0; i < output_cols_data.size(); i++)
    {
      EXPECT_EQ(int(output_cols_data[i].size()), output_tables[i].num_columns());
      for (int c = 0; c < table_n_cols; c++)
      {
        EXPECT_EQ(int(output_cols_data[i][c].size()), output_tables[i].num_rows());
        EXPECT_EQ(int(output_cols_null_count[i][c]), output_tables[i].get_column(c)->null_count);
        EXPECT_TRUE(
            cudf::test::column_wrapper<TypeParam>(output_cols_data[i][c],
                                                  output_cols_bitmask[i][c]) ==
            cudf::test::column_wrapper<TypeParam>(*output_tables[i].get_column(c)));
      }
    }
  }
}
