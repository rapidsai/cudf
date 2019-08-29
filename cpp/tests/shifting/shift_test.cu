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
 
 class ShiftTest : public GdfTest {};
 
 TEST_F(ShiftTest, positive)
 {
    auto fill_value = -9000;
    auto source0 = column_wrapper<int32_t>{ 5, 3, 1, 10 };
    auto source1 = column_wrapper<int32_t>{ 7, 4, 8, 12 };

    auto expect0 = column_wrapper<int32_t>{ fill_value, 5, 3, 1, };
    auto expect1 = column_wrapper<int32_t>{ fill_value, 7, 4, 8 };

    cudf::table source{source0.get(), source1.get()};
    cudf::table expect{expect0.get(), expect1.get()};
    cudf::table shifted = cudf::copy(source);

    cudf::shift(&shifted, source, 1, scalar_wrapper<int32_t>{fill_value, true});

    ASSERT_EQ(expect0, *shifted.get_column(0));
    ASSERT_EQ(expect1, *shifted.get_column(1));
 }

 TEST_F(ShiftTest, negative)
 {
    auto fill_value = -9000;
    auto source0 = column_wrapper<int32_t>{ 5, 3, 1, 10 };
    auto source1 = column_wrapper<int32_t>{ 7, 4, 8, 12 };

    auto expect0 = column_wrapper<int32_t>{ 3, 1, 10, fill_value };
    auto expect1 = column_wrapper<int32_t>{ 4, 8, 12, fill_value };

    cudf::table source{source0.get(), source1.get()};
    cudf::table expect{expect0.get(), expect1.get()};
    cudf::table shifted = cudf::copy(source);

    cudf::shift(&shifted, source, -1, scalar_wrapper<int32_t>{fill_value, true});

    ASSERT_EQ(expect0, *shifted.get_column(0));
    ASSERT_EQ(expect1, *shifted.get_column(1));
 }
