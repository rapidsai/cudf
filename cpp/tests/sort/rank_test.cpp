/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/table_utilities.hpp>
#include <vector>

void run_rank_test (cudf::table_view input,
                    cudf::table_view expected,
                    cudf::rank_method method,
                    cudf::order column_order,
                    cudf::include_nulls _include_nulls,
                    cudf::null_order null_precedence,
                    bool print=false
                    ) {
    // Rank
    auto got_rank_table = cudf::experimental::rank( input,
                                                    method,
                                                    column_order, 
                                                    _include_nulls, 
                                                    null_precedence);
    if(print) {
        cudf::test::print(got_rank_table->view().column(0)); std::cout<<"\n";
    }
    cudf::test::expect_tables_equal(expected, got_rank_table->view());
}

template <typename T>
struct Rank : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(Rank, cudf::test::NumericTypes);

TYPED_TEST(Rank, First)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T>   col1{{  5,   4,   3,   5,   8,   5}};
    cudf::test::fixed_width_column_wrapper<T>   col2{{  5,   4,   3,   5,   8,   5}, {1, 1, 0, 1, 1, 1}};
    cudf::test::strings_column_wrapper          col3({"d", "e", "a", "d", "k", "d"}, {1, 1, 1, 1, 1, 1});

    //FIRST
    //ASCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_asce_keep   {{3, 2, 1, 4, 6, 5}};
    cudf::test::fixed_width_column_wrapper<double> col2_asce_keep   {{2, 1, 6, 3, 5, 4}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double> col2_asce_top    {{3, 2, 1, 4, 6, 5}}; //BEFORE = TOP
    cudf::test::fixed_width_column_wrapper<double> col2_asce_bottom {{2, 1, 6, 3, 5, 4}}; //AFTER  = BOTTOM
    cudf::test::fixed_width_column_wrapper<double> col3_asce_keep   {{2, 5, 1, 3, 6, 4}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_asce_keep2  {{2, 5, 1, 3, 6, 4}};
    //DESCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_desc_keep   {{2, 5, 6, 3, 1, 4}};
    cudf::test::fixed_width_column_wrapper<double> col2_desc_keep   {{3, 6, 1, 4, 2, 5}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double> col2_desc_bottom {{2, 5, 6, 3, 1, 4}}; //BEFORE = BOTTOM
    cudf::test::fixed_width_column_wrapper<double> col2_desc_top    {{3, 6, 1, 4, 2, 5}}; //AFTER  = TOP
    cudf::test::fixed_width_column_wrapper<double> col3_desc_keep   {{3, 2, 6, 4, 1, 5}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_desc_keep2  {{3, 2, 6, 4, 1, 5}};
    auto first{cudf::rank_method::FIRST};

    // Rank
    if (std::is_same<T, cudf::experimental::bool8>::value) return;
    // Single column, Ascending
    //Non-null column
    run_rank_test(  cudf::table_view{{col1}}, cudf::table_view{{col1_asce_keep}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER);
    //Null column
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_asce_keep}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_asce_top}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //TOP
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_asce_bottom}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //BOTTOM
    // Single column, Descending
    //Non-null column
    run_rank_test(  cudf::table_view{{col1}}, cudf::table_view{{col1_desc_keep}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER);
    //Null column
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_desc_keep}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_desc_bottom}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //BOTTOM
    run_rank_test(  cudf::table_view{{col2}}, cudf::table_view{{col2_desc_top}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //TOP
    // Table, Ascending
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_asce_keep, col2_asce_keep}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_asce_keep, col2_asce_top}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //TOP
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_asce_keep, col2_asce_bottom}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //BOTTOM
    // Table, Descending           
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_desc_keep, col2_desc_keep}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_desc_keep, col2_desc_bottom}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //BOTTOM
    run_rank_test(  cudf::table_view{{col1, col2}}, cudf::table_view{{col1_desc_keep, col2_desc_top}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //TOP
    // Table with String column, Ascending
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_asce_keep, col2_asce_keep, col3_asce_keep}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_asce_keep, col2_asce_top, col3_asce_keep2}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //TOP
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_asce_keep, col2_asce_bottom, col3_asce_keep2}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //BOTTOM
    // Table with String column, Descending
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_desc_keep, col2_desc_keep, col3_desc_keep}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER); //KEEP
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_desc_keep, col2_desc_bottom, col3_desc_keep2}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::BEFORE); //BOTTOM
    run_rank_test(  cudf::table_view{{col1, col2, col3}}, cudf::table_view{{col1_desc_keep, col2_desc_top, col3_desc_keep2}}, first,
                    cudf::order::DESCENDING, 
                    cudf::include_nulls::YES, cudf::null_order::AFTER); //TOP
}

TYPED_TEST(Rank, Dense)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T>   col1{{  5,   4,   3,   5,   8,   5}};
    cudf::test::fixed_width_column_wrapper<T>   col2{{  5,   4,   3,   5,   8,   5}, {1, 1, 0, 1, 1, 1}};
    cudf::test::strings_column_wrapper          col3({"d", "e", "a", "d", "k", "d"}, {1, 1, 1, 1, 1, 1});
    //                                                  3    2    1   3    4     3

    //FIRST
    //ASCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_asce_keep   {{3, 2, 1, 3, 4, 3}};
    cudf::test::fixed_width_column_wrapper<double> col2_asce_keep   {{2, 1, 6, 3, 5, 4}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double> col2_asce_top    {{3, 2, 1, 4, 6, 5}}; //BEFORE = TOP
    cudf::test::fixed_width_column_wrapper<double> col2_asce_bottom {{2, 1, 6, 3, 5, 4}}; //AFTER  = BOTTOM
    cudf::test::fixed_width_column_wrapper<double> col3_asce_keep   {{2, 5, 1, 3, 6, 4}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_asce_keep2  {{2, 5, 1, 3, 6, 4}};
    //DESCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_desc_keep   {{2, 5, 6, 3, 1, 4}};
    cudf::test::fixed_width_column_wrapper<double> col2_desc_keep   {{3, 6, 1, 4, 2, 5}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double> col2_desc_bottom {{2, 5, 6, 3, 1, 4}}; //BEFORE = BOTTOM
    cudf::test::fixed_width_column_wrapper<double> col2_desc_top    {{3, 6, 1, 4, 2, 5}}; //AFTER  = TOP
    cudf::test::fixed_width_column_wrapper<double> col3_desc_keep   {{3, 2, 6, 4, 1, 5}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_desc_keep2  {{3, 2, 6, 4, 1, 5}};
    auto first{cudf::rank_method::DENSE};

    // Rank
    if (std::is_same<T, cudf::experimental::bool8>::value) return;
    // Single column, Ascending
    //Non-null column
    run_rank_test(  cudf::table_view{{col1}}, cudf::table_view{{col1_asce_keep}}, first,
                    cudf::order::ASCENDING, 
                    cudf::include_nulls::NO,  cudf::null_order::AFTER, true);
}
