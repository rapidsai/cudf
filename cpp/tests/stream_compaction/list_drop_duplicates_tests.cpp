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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

struct ListDropDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(ListDropDuplicatesTest, BasicList)
{
  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  using icw = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // clang-format off
  auto const idx = icw{ 0,  0,   1,      2,   1,      3,      4,   5,   5,      6,      4,     4 };
  auto const col = lcw{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input = cudf::table_view({idx, col});

  auto const exp_idx = icw{0, 1, 2, 3, 4, 5, 6};
  auto const exp_val = lcw{{}, {1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
  auto const expect  = cudf::table_view({exp_idx, exp_val});

  auto result        = cudf::distinct(input, {1});
  auto sorted_result = cudf::sort_by_key(*result, result->select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect, *sorted_result);
}

// TEST_F(ListDropDuplicatesTest, DeepList)
// {
//   using lcw = cudf::test::lists_column_wrapper<uint64_t>;
//   lcw col{
//     {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
//     {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
//     {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
//     {{7, 8}, {}},
//     lcw{lcw{}, lcw{}, lcw{}},
//     lcw{lcw{}},
//     lcw{lcw{}},
//     lcw{lcw{}},
//     lcw{lcw{}, lcw{}, lcw{}},
//     lcw{lcw{}, lcw{}, lcw{}},
//     {lcw{10}},
//     {lcw{10}},
//     {{13, 14}, {15}},
//     {{13, 14}, {16}},
//     lcw{},
//     lcw{lcw{}},
//   };

//   auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
//     1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11};
//   this->test_ungrouped_rank_scan(
//     col, expected_dense_vals, cudf::make_dense_rank_aggregation(), cudf::null_policy::INCLUDE);
// }

// TEST_F(ListDropDuplicatesTest, test)
// {
//   data_profile table_data_profile;
//   table_data_profile.set_distribution_params(cudf::type_id::LIST, distribution_id::UNIFORM, 0,
//   5); table_data_profile.set_null_frequency(0); auto const tbl =
//   create_random_table({cudf::type_id::LIST}, 1, row_count{10}, table_data_profile);
//   cudf::test::print(tbl->get_column(0));
//   auto const new_tbl = cudf::repeat(tbl->view(), 2);
//   cudf::test::print(new_tbl->get_column(0));
//   auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
//     1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
//   this->test_ungrouped_rank_scan(new_tbl->get_column(0),
//                                  expected_dense_vals,
//                                  cudf::make_dense_rank_aggregation(),
//                                  cudf::null_policy::INCLUDE);
// }

CUDF_TEST_PROGRAM_MAIN()
