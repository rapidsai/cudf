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
#include <tuple>


namespace cudf {
namespace test {

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
                                                    null_precedence,
                                                    false);
    if(print) {
        cudf::test::print(got_rank_table->view().column(0)); std::cout<<"\n";
    }
    cudf::test::expect_tables_equal(expected, got_rank_table->view());
}

using input_arg_t = std::tuple<cudf::order, cudf::include_nulls, cudf::null_order>;
input_arg_t asce_keep{cudf::order::ASCENDING, cudf::include_nulls::NO,  cudf::null_order::AFTER};
input_arg_t asce_top{cudf::order::ASCENDING, cudf::include_nulls::YES, cudf::null_order::BEFORE};
input_arg_t asce_bottom{cudf::order::ASCENDING, cudf::include_nulls::YES, cudf::null_order::AFTER};

input_arg_t desc_keep{cudf::order::DESCENDING, cudf::include_nulls::NO,  cudf::null_order::BEFORE};
input_arg_t desc_top{cudf::order::DESCENDING, cudf::include_nulls::YES, cudf::null_order::AFTER};
input_arg_t desc_bottom{cudf::order::DESCENDING, cudf::include_nulls::YES, cudf::null_order::BEFORE};
using test_case_t = std::tuple<input_arg_t, cudf::table_view, cudf::table_view>;
using table_view = cudf::table_view;

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
    cudf::test::fixed_width_column_wrapper<double>  col1_asce_keep   {{3, 2, 1, 4, 6, 5}};
    cudf::test::fixed_width_column_wrapper<double>& col1_asce_top    = col1_asce_keep;
    cudf::test::fixed_width_column_wrapper<double>& col1_asce_bottom = col1_asce_keep;
    cudf::test::fixed_width_column_wrapper<double>  col2_asce_keep   {{2, 1,-1, 3, 5, 4}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double>  col2_asce_top    {{3, 2, 1, 4, 6, 5}}; //BEFORE = TOP
    cudf::test::fixed_width_column_wrapper<double>  col2_asce_bottom {{2, 1, 6, 3, 5, 4}}; //AFTER  = BOTTOM
    cudf::test::fixed_width_column_wrapper<double>  col3_asce_keep   {{2, 5, 1, 3, 6, 4}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double>  col3_asce_top    {{2, 5, 1, 3, 6, 4}};
    cudf::test::fixed_width_column_wrapper<double>& col3_asce_bottom = col3_asce_top;
    //DESCENDING
    cudf::test::fixed_width_column_wrapper<double>  col1_desc_keep   {{2, 5, 6, 3, 1, 4}};
    cudf::test::fixed_width_column_wrapper<double>& col1_desc_top    = col1_desc_keep;
    cudf::test::fixed_width_column_wrapper<double>& col1_desc_bottom = col1_desc_keep;
    cudf::test::fixed_width_column_wrapper<double>  col2_desc_keep   {{2, 5,-1, 3, 1, 4}, {1, 1, 0, 1, 1, 1}}; //KEEP
    cudf::test::fixed_width_column_wrapper<double>  col2_desc_bottom {{2, 5, 6, 3, 1, 4}}; //BEFORE = BOTTOM
    cudf::test::fixed_width_column_wrapper<double>  col2_desc_top    {{3, 6, 1, 4, 2, 5}}; //AFTER  = TOP
    cudf::test::fixed_width_column_wrapper<double>  col3_desc_keep   {{3, 2, 6, 4, 1, 5}, {1, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<double>  col3_desc_top    {{3, 2, 6, 4, 1, 5}};
    cudf::test::fixed_width_column_wrapper<double>& col3_desc_bottom = col3_desc_top;

    // Rank
    if (std::is_same<T, cudf::experimental::bool8>::value) return;
    for (auto const &test_case : {
        // Single column, Ascending
        //Non-null column
        test_case_t{asce_keep, table_view{{col1}}, table_view{{col1_asce_keep}}},
        //Null column
        test_case_t{asce_keep, table_view{{col2}}, table_view{{col2_asce_keep}}},
        test_case_t{asce_top, table_view{{col2}}, table_view{{col2_asce_top}}},
        test_case_t{asce_bottom, table_view{{col2}}, table_view{{col2_asce_bottom}}},
        // Single column, Descending
        //Non-null column
        test_case_t{desc_keep, table_view{{col1}}, table_view{{col1_desc_keep}}},
        //Null column
        test_case_t{desc_keep, table_view{{col2}}, table_view{{col2_desc_keep}}},
        test_case_t{desc_top, table_view{{col2}}, table_view{{col2_desc_top}}},
        test_case_t{desc_bottom, table_view{{col2}}, table_view{{col2_desc_bottom}}},
        // Table, Ascending
        test_case_t{asce_keep, table_view{{col1,col2}}, table_view{{col1_asce_keep, col2_asce_keep}}},
        test_case_t{asce_top, table_view{{col1, col2}}, table_view{{col1_asce_top, col2_asce_top}}},
        test_case_t{asce_bottom, table_view{{col1, col2}}, table_view{{col1_asce_bottom, col2_asce_bottom}}},
        // Table, Descending
        test_case_t{desc_keep, table_view{{col1,col2}}, table_view{{col1_desc_keep, col2_desc_keep}}},
        test_case_t{desc_top, table_view{{col1, col2}}, table_view{{col1_desc_top, col2_desc_top}}},
        test_case_t{desc_bottom, table_view{{col1, col2}}, table_view{{col1_desc_bottom, col2_desc_bottom}}},
        // Table with String column, Ascending
        test_case_t{asce_keep, table_view{{col1, col2, col3}}, table_view{{col1_asce_keep, col2_asce_keep, col3_asce_keep}}},
        test_case_t{asce_top, table_view{{col1, col2, col3}}, table_view{{col1_asce_top, col2_asce_top, col3_asce_top}}},
        test_case_t{asce_bottom, table_view{{col1, col2, col3}}, table_view{{col1_asce_bottom, col2_asce_bottom, col3_asce_bottom}}},
        // Table with String column, Descending
        test_case_t{desc_keep, table_view{{col1, col2, col3}}, table_view{{col1_desc_keep, col2_desc_keep, col3_desc_keep}}},
        test_case_t{desc_top, table_view{{col1, col2, col3}}, table_view{{col1_desc_top, col2_desc_top, col3_desc_top}}},
        test_case_t{desc_bottom, table_view{{col1, col2, col3}}, table_view{{col1_desc_bottom, col2_desc_bottom, col3_desc_bottom}}},
        }) {
      input_arg_t input_arg;
      cudf::table_view input, output;
      std::tie(input_arg, input, output) = test_case;

      run_rank_test(
          input, output, cudf::rank_method::FIRST,
          std::get<0>(input_arg), std::get<1>(input_arg), std::get<2>(input_arg));
    }
}

TYPED_TEST(Rank, Dense)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T>   col1{{  5,   4,   3,   5,   8,   5}};
    cudf::test::fixed_width_column_wrapper<T>   col2{{  5,   4,   3,   5,   8,   5}, {1, 1, 0, 1, 1, 1}};
    cudf::test::strings_column_wrapper          col3({"d", "e", "a", "d", "k", "d"}, {1, 1, 1, 1, 1, 1});
    //                                                  3    2    1   3    4     3
    //ASCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_asce_keep    {{3, 2, 1, 3, 4, 3} };
    cudf::test::fixed_width_column_wrapper<double> col1_asce_top     {{3, 2, 1, 3, 4, 3} };
    cudf::test::fixed_width_column_wrapper<double> col1_asce_bottom  {{3, 2, 1, 3, 4, 3} };
    cudf::test::fixed_width_column_wrapper<double> col2_asce_keep    {{2, 1,-1, 2, 3, 2} , {1, 1, 0, 1, 1, 1} };
    cudf::test::fixed_width_column_wrapper<double> col2_asce_top     {{3, 2, 1, 3, 4, 3} };
    cudf::test::fixed_width_column_wrapper<double> col2_asce_bottom  {{2, 1, 4, 2, 3, 2} };
    cudf::test::fixed_width_column_wrapper<double> col3_asce_keep    {{2, 3, 1, 2, 4, 2} , {1, 1, 1, 1, 1, 1} };
    cudf::test::fixed_width_column_wrapper<double> col3_asce_top     {{2, 3, 1, 2, 4, 2} };
    cudf::test::fixed_width_column_wrapper<double> col3_asce_bottom  {{2, 3, 1, 2, 4, 2} };
    //DESCENDING
    cudf::test::fixed_width_column_wrapper<double> col1_desc_keep    {{2, 3, 4, 2, 1, 2} };
    cudf::test::fixed_width_column_wrapper<double> col1_desc_top     {{2, 3, 4, 2, 1, 2} };
    cudf::test::fixed_width_column_wrapper<double> col1_desc_bottom  {{2, 3, 4, 2, 1, 2} };
    cudf::test::fixed_width_column_wrapper<double> col2_desc_keep    {{2, 3,-1, 2, 1, 2} , {1, 1, 0, 1, 1, 1} };
    cudf::test::fixed_width_column_wrapper<double> col2_desc_top     {{3, 4, 1, 3, 2, 3} };
    cudf::test::fixed_width_column_wrapper<double> col2_desc_bottom  {{2, 3, 4, 2, 1, 2} };
    cudf::test::fixed_width_column_wrapper<double> col3_desc_keep    {{3, 2, 4, 3, 1, 3} , {1, 1, 1, 1, 1, 1} };
    cudf::test::fixed_width_column_wrapper<double> col3_desc_top     {{3, 2, 4, 3, 1, 3} };
    cudf::test::fixed_width_column_wrapper<double> col3_desc_bottom  {{3, 2, 4, 3, 1, 3} };

    // Rank
    if (std::is_same<T, cudf::experimental::bool8>::value) return;
    for (auto const &test_case : {
        // Single column, Ascending
        //Non-null column
        test_case_t{asce_keep, table_view{{col1}}, table_view{{col1_asce_keep}}},
        // //Null column
        test_case_t{asce_keep, table_view{{col2}}, table_view{{col2_asce_keep}}},
        test_case_t{asce_top, table_view{{col2}}, table_view{{col2_asce_top}}},
        test_case_t{asce_bottom, table_view{{col2}}, table_view{{col2_asce_bottom}}},
        // Single column, Descending
        //Non-null column
        test_case_t{desc_keep, table_view{{col1}}, table_view{{col1_desc_keep}}},
        //Null column
        test_case_t{desc_keep, table_view{{col2}}, table_view{{col2_desc_keep}}},
        test_case_t{desc_top, table_view{{col2}}, table_view{{col2_desc_top}}},
        test_case_t{desc_bottom, table_view{{col2}}, table_view{{col2_desc_bottom}}},
        // Table, Ascending
        test_case_t{asce_keep, table_view{{col1,col2}}, table_view{{col1_asce_keep, col2_asce_keep}}},
        test_case_t{asce_top, table_view{{col1, col2}}, table_view{{col1_asce_top, col2_asce_top}}},
        test_case_t{asce_bottom, table_view{{col1, col2}}, table_view{{col1_asce_bottom, col2_asce_bottom}}},
        // Table, Descending
        test_case_t{desc_keep, table_view{{col1,col2}}, table_view{{col1_desc_keep, col2_desc_keep}}},
        test_case_t{desc_top, table_view{{col1, col2}}, table_view{{col1_desc_top, col2_desc_top}}},
        test_case_t{desc_bottom, table_view{{col1, col2}}, table_view{{col1_desc_bottom, col2_desc_bottom}}},
        // Table with String column, Ascending
        test_case_t{asce_keep, table_view{{col1, col2, col3}}, table_view{{col1_asce_keep, col2_asce_keep, col3_asce_keep}}},
        test_case_t{asce_top, table_view{{col1, col2, col3}}, table_view{{col1_asce_top, col2_asce_top, col3_asce_top}}},
        test_case_t{asce_bottom, table_view{{col1, col2, col3}}, table_view{{col1_asce_bottom, col2_asce_bottom, col3_asce_bottom}}},
        // Table with String column, Descending
        test_case_t{desc_keep, table_view{{col1, col2, col3}}, table_view{{col1_desc_keep, col2_desc_keep, col3_desc_keep}}},
        test_case_t{desc_top, table_view{{col1, col2, col3}}, table_view{{col1_desc_top, col2_desc_top, col3_desc_top}}},
        test_case_t{desc_bottom, table_view{{col1, col2, col3}}, table_view{{col1_desc_bottom, col2_desc_bottom, col3_desc_bottom}}},
        }) {
      input_arg_t input_arg;
      cudf::table_view input, output;
      std::tie(input_arg, input, output) = test_case;

      run_rank_test(
          input, output, cudf::rank_method::DENSE,
          std::get<0>(input_arg), std::get<1>(input_arg), std::get<2>(input_arg), false);
    }
}

} // namespace test
} // namespace cudf
