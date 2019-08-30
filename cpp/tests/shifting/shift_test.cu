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
 #include <cudf/shifting.hpp>
 
 using cudf::test::column_wrapper;
 using cudf::test::scalar_wrapper;

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
 
class ShiftTest : public GdfTest {};

TEST_F(ShiftTest, positive)
{
  auto source_column = make_column_wrapper<int32_t>(
    {9, 8, 7, 6, 5, 4, 3, 2, 1},
    {0, 1, 0, 1, 1, 1, 0, 1, 0}
  );

  auto expect_column = make_column_wrapper<int32_t>(
    {9, 8, 7, 8, 5, 6, 5, 4, 1},
    {0, 0, 0, 1, 0, 1, 1, 1, 0}
  );

  cudf::table source{source_column.get()};
  cudf::table expect{expect_column.get()};

  auto shifted = cudf::shift(source, 2, scalar_wrapper<int32_t>{0, false});

  auto actual_column = *shifted.get_column(0);

  print_gdf_column(source_column.get());
  print_gdf_column(expect_column.get());
  print_gdf_column(&actual_column);

  ASSERT_EQ(expect_column, *shifted.get_column(0));
}