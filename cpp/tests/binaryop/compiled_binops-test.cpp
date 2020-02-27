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

#include <iostream>
#include <vector>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/cudf.h>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf/compiled_binaryop.hpp>

#include <tests/utilities/column_utilities.hpp>
#include <thrust/device_vector.h>

//template <typename T>
struct BinaryOpsExpTest : public cudf::test::BaseFixture { };

//TYPED_TEST_CASE(BinaryOpsExpTest, cudf::test::NumericTypes);
//TYPED_TEST(BinaryOpsExpTest, Sum)
TEST_F(BinaryOpsExpTest, Sum)
{
    //using T = TypeParam;
    using T = int;
    std::vector<T> v1({1, 2, 3, 4, 0, -1, -2, -3});
    std::vector<T> v2({1, 2, 3, 4, 0,  1,  2,  3});
    std::vector<T> v3({2, 4, 6, 8, 0,  0,  0,  0});
    //std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});

    // test without nulls
    cudf::test::fixed_width_column_wrapper<T> lhs(v1.begin(), v1.end());
    cudf::test::fixed_width_column_wrapper<T> rhs(v2.begin(), v2.end());
    cudf::test::fixed_width_column_wrapper<T> expect(v3.begin(), v3.end());

    cudf::test::print(lhs); std::cout << "\n";
    cudf::test::print(rhs); std::cout << "\n";

    auto out1 = cudf::experimental::experimental_binary_operation1(lhs, rhs, 
        cudf::data_type(cudf::experimental::type_to_id<T>()));
    cudf::test::print(out1->view()); std::cout << "\n";
    auto out2 = cudf::experimental::experimental_binary_operation1(lhs, rhs, 
        cudf::data_type(cudf::experimental::type_to_id<T>()));
    cudf::test::print(out2->view()); std::cout << "\n";
    auto out3 = cudf::experimental::experimental_binary_operation1(lhs, rhs, 
        cudf::data_type(cudf::experimental::type_to_id<T>()));
    cudf::test::print(out3->view()); std::cout << "\n";

    // test with nulls
}

