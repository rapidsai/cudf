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

 #include <tests/utilities/cudf_test_fixtures.h>
 #include <tests/utilities/column_wrapper.cuh>
 #include <tests/utilities/scalar_wrapper.cuh>
 #include <cudf/copying.hpp>
 
 using cudf::test::column_wrapper;
 using cudf::test::scalar_wrapper;

 namespace
 {

 template <typename ColumnType>
 column_wrapper<ColumnType> make_column_wrapper(
   std::vector<ColumnType> data,
   std::vector<gdf_valid_type> mask
 )
 {
   return column_wrapper<ColumnType>(
     data,
     [mask](gdf_size_type row){ return mask[row]; }
   );
 }

 template <typename ColumnType>
 void test_shift(
   gdf_index_type periods,
   scalar_wrapper<ColumnType> fill_value,
   column_wrapper<ColumnType> source_column,
   column_wrapper<ColumnType> expect_column
 )
 {
   auto source_table = cudf::table{source_column.get()};
   auto actual_table = cudf::shift(source_table, periods, fill_value);
 
   print_gdf_column(source_column.get());
   print_gdf_column(expect_column.get());
   print_gdf_column(actual_table.get_column(0));

   std::cout << "a: " << expect_column.get()->null_count << std::endl;
   std::cout << "b: " << actual_table.get_column(0)->null_count << std::endl;

   auto actual_column = column_wrapper<ColumnType>(*actual_table.get_column(0));
 
   ASSERT_EQ(expect_column, actual_column);
 }

 
class ShiftTest : public GdfTest {};

TEST_F(ShiftTest, positive)
{
  auto source_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {1, 1, 1, 1, 1, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {0, 0, 0, 0, 0, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  test_shift(
    5,
    scalar_wrapper<int32_t>(5, false),
    source_column,
    expect_column
  );
}

TEST_F(ShiftTest, negative)
{
  auto source_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {1, 1, 1, 1, 1, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {0, 0, 0, 0, 0, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  test_shift(
    5,
    scalar_wrapper<int32_t>(5, false),
    source_column,
    expect_column
  );
}

TEST_F(ShiftTest, valid_fill)
{
  auto source_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {1, 1, 1, 1, 1, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 5, 5, 5, 5, 5, 5},
    {0, 0, 0, 0, 0, 1, 1, 1, 1}
    // {0, 0, 0, 0, 0, 0, 0, 0, 0}
  );

  test_shift(
    5,
    scalar_wrapper<int32_t>(5, false),
    source_column,
    expect_column
  );
}

// TEST_F(ShiftTest, zero_shift)
// {
//   auto source_column = make_column_wrapper<int32_t>(
//     {7, 7, 7, 7, 7, 7, 7, 7, 7},
//     {1, 1, 1, 1, 1, 1, 1, 1, 1}
//   );

//   auto expect_column = make_column_wrapper<int32_t>(
//     {7, 7, 7, 7, 7, 7, 7, 7, 7},
//     {1, 1, 1, 1, 1, 1, 1, 1, 1}
//   );

//   test_shift(0, scalar_wrapper<int32_t>(7, true), source_column, expect_column);
// }

}