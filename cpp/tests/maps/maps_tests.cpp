/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "cudf/column/column_factories.hpp"
#include "cudf/maps/maps_column_view.hpp"
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf::test
{

using structs = structs_column_wrapper;
template <typename T> using fwcw = fixed_width_column_wrapper<T, int32_t>;
template <typename T> using lists = lists_column_wrapper<T, int32_t>;
using offsets = fwcw<cudf::size_type>;

template <typename T> struct MapsTest : BaseFixture {};

using MapsTestTypes = Concat<IntegralTypesNotBool, FloatingPointTypes, ChronoTypes>;

TYPED_TEST_SUITE(MapsTest, MapsTestTypes);

TYPED_TEST(MapsTest, BasicConstruction)
{
    using T = TypeParam;

    auto const list_binary_structs = [] {
        auto keys = fwcw<T>{0,1,2, 3,0,4, 5,6,0, 7,8,9};
        auto vals = fwcw<T>{0,1,2, 3,4,5, 6,7,8, 9,10,11};
        auto key_vals = structs{keys, vals};

        return cudf::make_lists_column(4, offsets{0, 3, 6, 9, 12}.release(), key_vals.release(), 0, {});
    }();

    auto const maps = cudf::maps_column_view{list_binary_structs->view()};
    auto const expected_keys = lists<T>{{0,1,2}, {3,0,4}, {5,6,0}, {7,8,9}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.keys().parent(), expected_keys);
    auto const expected_values = lists<T>{{0,1,2}, {3,4,5}, {6,7,8}, {9,10,11}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.values().parent(), expected_values);
}

TYPED_TEST(MapsTest, SlicedConstruction)
{
    using T = TypeParam;

    auto const list_binary_structs_column = [] {
        auto keys = fwcw<T>{0,0,0, 0,1,2, 3,0,4, 5,6,0, 7,8,9, 9,9,9};
        auto vals = fwcw<T>{0,0,0, 0,1,2, 3,4,5, 6,7,8, 9,10,11, 9,9,9};
        auto key_vals = structs{keys, vals};

        return cudf::make_lists_column(6, offsets{0, 3, 6, 9, 12, 15, 18}.release(), key_vals.release(), 0, {});
    }();

    auto const lists_view = list_binary_structs_column->view();
    auto const sliced_lists = cudf::slice(cudf::table_view{{lists_view}}, {1,5}).front().column(0);
    auto const maps = cudf::maps_column_view{sliced_lists};
    auto const expected_keys = lists<T>{{0,1,2}, {3,0,4}, {5,6,0}, {7,8,9}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.keys().parent(), expected_keys);
    auto const expected_values = lists<T>{{0,1,2}, {3,4,5}, {6,7,8}, {9,10,11}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.values().parent(), expected_values);
}

TYPED_TEST(MapsTest, BasicLookup)
{
    using T = TypeParam;

    auto const list_binary_structs_column = [] {
        auto keys = fwcw<T>{0,0,0, 0,1,2, 3,0,4, 5,6,0, 7,8,9, 9,9,9};
        auto vals = fwcw<T>{0,0,0, 0,1,2, 3,4,5, 6,7,8, 9,10,11, 9,9,9};
        auto key_vals = structs{keys, vals};

        return cudf::make_lists_column(6, offsets{0, 3, 6, 9, 12, 15, 18}.release(), key_vals.release(), 0, {});
    }();

    auto const lists_view = list_binary_structs_column->view();
    auto const maps = cudf::maps_column_view{lists_view};
    auto const lookup_result = maps.get_values_for(fwcw<T>{0, 0, 0, 0, 0, 0});
    print(lookup_result->view());
}



} // namespace cudf::test;
