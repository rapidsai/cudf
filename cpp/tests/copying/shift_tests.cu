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
 cudf::test::column_wrapper<ColumnType> make_column_wrapper(
   std::vector<ColumnType> data,
   std::vector<gdf_valid_type> mask
 )
 {
   return cudf::test::column_wrapper<ColumnType>(
     data,
     [mask](gdf_size_type row){ return mask[row]; }
   );
 }

 template <typename ColumnType>
 void test_shift(
   gdf_index_type period,
   scalar_wrapper<ColumnType> fill_value,
   column_wrapper<ColumnType> source_column,
   column_wrapper<ColumnType> expect_column
 )
 {
   auto actual_column = cudf::shift(source_column, period, fill_value);
 
   print_gdf_column(source_column.get());
   print_gdf_column(expect_column.get());
   print_gdf_column(&actual_column);
 
   ASSERT_EQ(expect_column, actual_column);
 }

 
class ShiftTest : public GdfTest {};

TEST_F(ShiftTest, positive)
{
  auto source_column = make_column_wrapper<int32_t>(
    {9, 8, 7, 6, 5, 4, 3, 2, 1},
    {0, 1, 0, 1, 1, 1, 0, 1, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {0, 0, 9, 8, 7, 6, 5, 4, 3},
    {0, 0, 0, 1, 0, 1, 1, 1, 0}
  );

  test_shift(
    2,
    scalar_wrapper<int32_t>(0, false),
    source_column,
    expect_column
  );
}

TEST_F(ShiftTest, negative)
{
  auto source_column = make_column_wrapper<int32_t>(
    {9, 8, 7, 6, 5, 4, 3, 2, 1},
    {0, 1, 0, 1, 1, 1, 0, 1, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {7, 6, 5, 4, 3, 2, 1, 0, 0},
    {0, 1, 1, 1, 0, 1, 0, 0, 0}
  );

  test_shift(-2, scalar_wrapper<int32_t>(0, false), source_column, expect_column);
}

TEST_F(ShiftTest, valid_fill)
{
  auto source_column = make_column_wrapper<int32_t>(
    {9, 8, 7, 6, 5, 4, 3, 2, 1},
    {0, 1, 0, 1, 1, 1, 0, 1, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {5, 5, 5, 9, 8, 7, 6, 5, 4},
    {1, 1, 1, 0, 1, 0, 1, 1, 1}
  );

  test_shift(3, scalar_wrapper<int32_t>{5, true}, source_column, expect_column);
}

}