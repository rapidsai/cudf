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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
//TODO remove after PR 3490 merge
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>

#include <thrust/device_vector.h>

using scan_op = cudf::experimental::scan_op;
using cudf::column_view;

// This is the main test feature
template <typename T>
struct ScanTest : public cudf::test::BaseFixture
{
    void scan_test(
        cudf::test::fixed_width_column_wrapper<T> const col_in,
        cudf::test::fixed_width_column_wrapper<T> const expected_col_out,
        scan_op op, bool inclusive)
    {
        bool do_print = false;

        auto int_values = cudf::test::to_host<T>(col_in);
        auto exact_values = cudf::test::to_host<T>(expected_col_out);
        this->val_check(std::get<0>(int_values), do_print, "input = ");
        this->val_check(std::get<0>(exact_values), do_print, "exact = ");

        const column_view input_view = col_in;
        std::unique_ptr<cudf::column> col_out;

        CUDF_EXPECT_NO_THROW( col_out = cudf::experimental::scan(input_view, op, inclusive) );
        const column_view result_view = col_out->view();

        cudf::test::expect_column_properties_equal(input_view, result_view);
        cudf::test::expect_columns_equal(expected_col_out, result_view);

        auto host_result = cudf::test::to_host<T>(result_view);
        this->val_check(std::get<0>(host_result), do_print, "result = ");
    }

    template <typename Ti>
    void val_check(std::vector<Ti> const & v, bool do_print=false, const char* msg = nullptr){
        if( do_print ){
            std::cout << msg << " {";
            std::for_each(v.begin(), v.end(), [](Ti i){ std::cout << ", " <<  i;});
            std::cout << "}"  << std::endl;
        }
        range_check(v);
    }

    // make sure all elements in the range of sint8([-128, 127])
    template <typename Ti>
    void range_check(std::vector<Ti> const & v){
        std::for_each(v.begin(), v.end(),
            [](Ti i){
                ASSERT_GE(static_cast<int>(i), -128);
                ASSERT_LT(static_cast<int>(i),  128);
            });
    }

};

using Types = cudf::test::NumericTypes;
//using Types = testing::Types<int32_t>;

TYPED_TEST_CASE(ScanTest, Types);

// ------------------------------------------------------------------------
TYPED_TEST(ScanTest, Min)
{
    std::vector<TypeParam>  v({123, 64, 63, 99, -5, 123, -16, -120, -111});
    std::vector<bool> b({  1,  0,  1,  1,  1,   1,   0,    1,    1});
    std::vector<TypeParam> exact(v.size());

    std::transform(v.cbegin(), v.cend(),
        exact.begin(),
        [acc=v[0]](auto i) mutable { acc = std::min(acc, i); return acc; }
        );

    this->scan_test({v.begin(), v.end()}, 
                    {exact.begin(), exact.end()},
                    scan_op::MIN, true);

    std::transform(v.cbegin(), v.cend(), b.begin(),
        exact.begin(),
        [acc=v[0]](auto i, bool b) mutable { if(b) acc = std::min(acc, i); return acc; }
        );

    this->scan_test({v.begin(), v.end(), b.begin()}, 
                    {exact.begin(), exact.end(), b.begin()},
                    scan_op::MIN, true);
}

TYPED_TEST(ScanTest, Max)
{
    std::vector<TypeParam>  v({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});
    std::vector<bool> b({   1, 0, 1,    1,    1,  1,  0,  1,   1,   1});
    std::vector<TypeParam> exact(v.size());

    std::transform(v.cbegin(), v.cend(),
        exact.begin(),
        [acc=v[0]](auto i) mutable { acc = std::max(acc, i); return acc; }
        );

    this->scan_test({v.begin(), v.end()}, 
                    {exact.begin(), exact.end()},
                    scan_op::MAX, true);

    std::transform(v.cbegin(), v.cend(), b.begin(),
        exact.begin(),
        [acc=v[0]](auto i, bool b) mutable { if(b) acc = std::max(acc, i); return acc; }
        );

    this->scan_test({v.begin(), v.end(), b.begin()}, 
                    {exact.begin(), exact.end(), b.begin()},
                    scan_op::MAX, true);
}


TYPED_TEST(ScanTest, Product)
{
    std::vector<TypeParam>  v({5, -1, 1, 3, -2, 4});
    std::vector<bool> b({1,  1, 1, 0,  1, 1});
    std::vector<TypeParam> exact(v.size());

    std::transform(v.cbegin(), v.cend(),
        exact.begin(),
        [acc=1](auto i) mutable { acc *= i; return acc; }
        );

    this->scan_test({v.begin(), v.end()}, 
                    {exact.begin(), exact.end()},
                    scan_op::PRODUCT, true);

    std::transform(v.cbegin(), v.cend(), b.begin(),
        exact.begin(),
        [acc=1](auto i, bool b) mutable { if(b) acc *= i; return acc; }
        );

    this->scan_test({v.begin(), v.end(), b.begin()}, 
                    {exact.begin(), exact.end(), b.begin()},
                    scan_op::PRODUCT, true);
}

TYPED_TEST(ScanTest, Sum)
{
    std::vector<TypeParam>  v({-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});
    std::vector<bool> b({   1, 0, 1,   1,    0,  0,   1, 1,  1,   1});
    std::vector<TypeParam> exact(v.size());

    std::transform(v.cbegin(), v.cend(),
        exact.begin(),
        [acc=0](auto i) mutable { acc += i; return acc; }
        );

    this->scan_test({v.begin(), v.end()}, 
                    {exact.begin(), exact.end()},
                    scan_op::SUM, true);

    std::transform(v.cbegin(), v.cend(), b.begin(),
        exact.begin(),
        [acc=0](auto i, bool b) mutable { if(b) acc += i; return acc; }
        );

    this->scan_test({v.begin(), v.end(), b.begin()}, 
                    {exact.begin(), exact.end(), b.begin()},
                    scan_op::SUM, true);
}
