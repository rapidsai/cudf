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

namespace cudf {
namespace test {

struct ListRankScanTest : public cudf::test::BaseFixture {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       std::unique_ptr<aggregation> const& agg,
                                       null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, col_out->view(), debug_output_level::ALL_ERRORS);
  }
};

TEST_F(ListRankScanTest, BasicList)
{
  using lcw      = cudf::test::lists_column_wrapper<uint64_t>;
  auto const col = lcw{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9};
  this->test_ungrouped_rank_scan(
    col, expected_dense_vals, cudf::make_dense_rank_aggregation(), null_policy::INCLUDE);
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
    col, expected_dense_vals, cudf::make_dense_rank_aggregation(), null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, test)
{
  data_profile table_data_profile;
  table_data_profile.set_distribution_params(type_id::LIST, distribution_id::UNIFORM, 0, 5);
  table_data_profile.set_null_frequency(0);
  auto const tbl = create_random_table({type_id::LIST}, 1, row_count{10}, table_data_profile);
  print(tbl->get_column(0));
  auto const new_tbl = cudf::repeat(tbl->view(), 2);
  print(new_tbl->get_column(0));
  auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
  this->test_ungrouped_rank_scan(new_tbl->get_column(0),
                                 expected_dense_vals,
                                 cudf::make_dense_rank_aggregation(),
                                 null_policy::INCLUDE);
}
}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
