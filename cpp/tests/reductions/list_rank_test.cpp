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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include "benchmarks/common/generate_input.hpp"
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>

struct ListRankScanTest : public cudf::test::BaseFixture {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       std::unique_ptr<cudf::aggregation> const& agg,
                                       cudf::null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, cudf::scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, col_out->view(), cudf::test::debug_output_level::ALL_ERRORS);
  }
};

TEST_F(ListRankScanTest, BasicList)
{
  using lcw      = cudf::test::lists_column_wrapper<uint64_t>;
  auto const col = lcw{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9};
  this->test_ungrouped_rank_scan(
    col, expected_dense_vals, cudf::make_dense_rank_aggregation(), cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, DeepList)
{
  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
    {{7, 8}, {}},
    lcw{lcw{}, lcw{}, lcw{}},
    lcw{lcw{}},
    lcw{lcw{}},
    lcw{lcw{}},
    lcw{lcw{}, lcw{}, lcw{}},
    lcw{lcw{}, lcw{}, lcw{}},
    {lcw{10}},
    {lcw{10}},
    {{13, 14}, {15}},
    {{13, 14}, {16}},
    lcw{},
    lcw{lcw{}},
  };

  auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11};
  this->test_ungrouped_rank_scan(
    col, expected_dense_vals, cudf::make_dense_rank_aggregation(), cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, Datagen)
{
  data_profile table_data_profile;
  table_data_profile.set_distribution_params(cudf::type_id::LIST, distribution_id::UNIFORM, 0, 5);
  table_data_profile.set_null_frequency(0);
  auto const tbl = create_random_table({cudf::type_id::LIST}, 1, row_count{10}, table_data_profile);
  cudf::test::print(tbl->get_column(0));
  auto const new_tbl = cudf::repeat(tbl->view(), 2);
  cudf::test::print(new_tbl->get_column(0));
  auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
  this->test_ungrouped_rank_scan(new_tbl->get_column(0),
                                 expected_dense_vals,
                                 cudf::make_dense_rank_aggregation(),
                                 cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, ListStruct)
{
  // Constructing a list of struct of two elements
  // []                  ==
  // []                  !=
  // Null                ==
  // Null                !=
  // [Null, Null]        !=
  // [Null]              ==
  // [Null]              ==
  // [Null]              !=
  // [{Null, Null}]      !=
  // [{1,'a'}, {2,'b'}]  !=
  // [{0,'a'}, {2,'b'}]  !=
  // [{0,'a'}, {2,'c'}]  ==
  // [{0,'a'}, {2,'c'}]  !=
  // [{0,Null}]          ==
  // [{0,Null}]          !=
  // [{Null, 0}]         ==
  // [{Null, 0}]

  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{
    {-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::strings_column_wrapper{
    {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto struc = cudf::test::structs_column_wrapper{
    {col1, col2}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto nullmask_buf =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                       17,
                                       nullptr,
                                       static_cast<cudf::bitmask_type*>(nullmask_buf.data()),
                                       cudf::UNKNOWN_NULL_COUNT,
                                       0,
                                       {offsets, struc});

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};

  this->test_ungrouped_rank_scan(
    list_column, expect, cudf::make_dense_rank_aggregation(), cudf::null_policy::INCLUDE);
}

CUDF_TEST_PROGRAM_MAIN()
