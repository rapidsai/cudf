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

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/legacy/interop.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <vector>

template <typename T>
struct SortedOrder : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SortedOrder, cudf::test::NumericTypes);

TYPED_TEST(SortedOrder, WithNullMax)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k", "d"}, {1, 1, 0, 1, 1, 1});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 1}};
    cudf::table_view input {{col1, col2, col3}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{1, 0, 5, 3, 4, 2}};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order> null_precedence {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER};


    auto got = cudf::experimental::sorted_order(input, column_order, null_precedence);

    if (!std::is_same<T, cudf::experimental::bool8>::value) {
        cudf::test::expect_columns_equal(expected, got->view());
    } else {
        // for bools only validate that the null element landed at the back, since
        // the rest of the values are equivalent and yields random sorted order.
        auto to_host = [](cudf::column_view const& col) {
            std::vector<int32_t> h_data(col.size());
            cudaMemcpy(h_data.data(), col.data<int32_t>(),
                       h_data.size() * sizeof(int32_t),
                       cudaMemcpyDefault);
            return h_data;
        };
        std::vector<int32_t> h_exp = to_host(expected);
        std::vector<int32_t> h_got = to_host(got->view());
        EXPECT_EQ(h_exp.at(h_exp.size() - 1),
                  h_got.at(h_got.size() - 1));
    }
}

TYPED_TEST(SortedOrder, WithNullMin)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {1, 1, 0, 1, 1}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"}, {1, 1, 0, 1, 1});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 1, 0, 1, 1}};
    cudf::table_view input {{col1, col2, col3}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};

    auto got = cudf::experimental::sorted_order(input, column_order);

    if (!std::is_same<T, cudf::experimental::bool8>::value) {
        cudf::test::expect_columns_equal(expected, got->view());
    } else {
        // for bools only validate that the null element landed at the front, since
        // the rest of the values are equivalent and yields random sorted order.
        auto to_host = [](cudf::column_view const& col) {
            std::vector<int32_t> h_data(col.size());
            cudaMemcpy(h_data.data(), col.data<int32_t>(),
                       h_data.size() * sizeof(int32_t),
                       cudaMemcpyDefault);
            return h_data;
        };
        std::vector<int32_t> h_exp = to_host(expected);
        std::vector<int32_t> h_got = to_host(got->view());
        EXPECT_EQ(h_exp.at(0),
                  h_got.at(0));
    }
}

TYPED_TEST(SortedOrder, WithMixedNullOrder)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8},    {0, 0, 1, 1, 0}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"}, {0, 1, 0, 0, 1});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 0, 1, 0, 1}};
    cudf::table_view input {{col1, col2, col3}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 3, 0, 1, 4}};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence {cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

    auto got = cudf::experimental::sorted_order(input, column_order, null_precedence);

    if (!std::is_same<T, cudf::experimental::bool8>::value) {
        cudf::test::expect_columns_equal(expected, got->view());
    } else {
        // for bools only validate that the null element landed at the front, since
        // the rest of the values are equivalent and yields random sorted order.
        auto to_host = [](cudf::column_view const& col) {
            std::vector<int32_t> h_data(col.size());
            cudaMemcpy(h_data.data(), col.data<int32_t>(),
                       h_data.size() * sizeof(int32_t),
                       cudaMemcpyDefault);
            return h_data;
        };
        std::vector<int32_t> h_exp = to_host(expected);
        std::vector<int32_t> h_got = to_host(got->view());
        EXPECT_EQ(h_exp.at(0),
                  h_got.at(0));
    }
}

TYPED_TEST(SortedOrder, WithAllValid)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
    cudf::table_view input {{col1, col2, col3}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};

    auto got = cudf::experimental::sorted_order(input, column_order);

    // Skip validating bools order. Valid true bools are all
    // equivalent, and yield random order after thrust::sort
    if (!std::is_same<T, cudf::experimental::bool8>::value) {
        cudf::test::expect_columns_equal(expected, got->view());
    }
}

TYPED_TEST(SortedOrder, MisMatchInColumnOrderSize)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
    cudf::table_view input {{col1, col2, col3}};

    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING};

    EXPECT_THROW(cudf::experimental::sorted_order(input, column_order), cudf::logic_error);
}

TYPED_TEST(SortedOrder, MisMatchInNullPrecedenceSize)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
    cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
    cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
    cudf::table_view input {{col1, col2, col3}};

    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order>  null_precedence {cudf::null_order::AFTER, cudf::null_order::BEFORE};

    EXPECT_THROW(cudf::experimental::sorted_order(input, column_order, null_precedence), cudf::logic_error);
}

TYPED_TEST(SortedOrder, ZeroSizedColumns)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> col1{};
    cudf::table_view input {{col1}};

    cudf::test::fixed_width_column_wrapper<int32_t> expected{};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};

    auto got = cudf::experimental::sorted_order(input, column_order);

    cudf::test::expect_columns_equal(expected, got->view());
}
