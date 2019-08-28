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
 #include <cudf/copying.hpp>
 #include <cudf/shift.hpp>
 
 using cudf::test::column_wrapper;
 
 class ShiftTest : public GdfTest {};
 
 TEST_F(ShiftTest, empty_table)
 {
    // std::vector<int32_t> column_0_data  {  10, 20, 30, 40 };
    // std::vector<bool>    column_0_valid {   1,  1,  1,  1 };
    // std::vector<int32_t> column_1_data  {  50, 60, 70, 80 };
    // std::vector<bool>    column_2_valid {   1,  1,  1,  1 };

    // auto column_0 = column_wrapper<int32_t> ( column_0_data,
    //     [&]( gdf_index_type row ) { return column_0_valid[row]; }
    // );

    // auto column_1 = column_wrapper<int32_t> ( column_1_data,
    //     [&]( gdf_index_type row ) { return column_1_valid[row]; }
    // );

    // std::vector<gdf_column*> columns { column_0.get(), column_1.get() };

    // auto input_table = cudf::table(columns)

    auto source0 = column_wrapper<int32_t>{ 5, 3, 1, 10 };
    auto source1 = column_wrapper<int32_t>{ 7, 4, 8, 12 };

    auto expect0 = column_wrapper<int32_t>{ 5, 3, 1, 10 };
    auto expect1 = column_wrapper<int32_t>{ 7, 4, 8, 12 };

    cudf::table source{source0.get(), source1.get()};
    cudf::table expect{expect0.get(), expect1.get()};
    cudf::table shifted = cudf::copy(source);

    //// b
    cudf::shift();
    //// e

    ASSERT_EQ(expect0, *shifted.get_column(0));
    ASSERT_EQ(expect1, *shifted.get_column(1));
 }
