/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <limits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct UniqueJoinTest : public cudf::test::BaseFixture {};

TEST_F(UniqueJoinTest, IntegerInnerJoin)
{
  auto constexpr num = 2024;

  auto const init = cudf::numeric_scalar<int32_t>{0};

  auto build = cudf::sequence(num, init, cudf::numeric_scalar<int32_t>{1});
  auto probe = cudf::sequence(num, init, cudf::numeric_scalar<int32_t>{2});

  auto unique_join = cudf::unique_hash_join<cudf::has_nested::NO>{
    cudf::table_view{{build->view()}}, cudf::table_view{{probe->view()}}, cudf::nullable_join::NO};

  auto result = unique_join.inner_join();

  auto constexpr gold_size = num / 2;
  EXPECT_EQ(result.first->size(), gold_size);
  EXPECT_EQ(result.second->size(), gold_size);
}

TEST_F(UniqueJoinTest, InnerJoinNestedNoNulls)
{
  column_wrapper<int32_t> col0_0{{1, 2, 3, 4, 5}};
  strcol_wrapper col0_1({"s0", "s0", "s3", "s4", "s5"});
  column_wrapper<int32_t> col0_2{{9, 9, 9, 9, 9}};

  column_wrapper<int32_t> col1_0{{1, 2, 3, 4, 9}};
  strcol_wrapper col1_1({"s0", "s0", "s0", "s4", "s4"});
  column_wrapper<int32_t> col1_2{{9, 9, 9, 0, 9}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table build(std::move(cols0));
  Table probe(std::move(cols1));

  auto unique_join = cudf::unique_hash_join<cudf::has_nested::YES>{build.view(), probe.view()};
  auto [left_join_indices, right_join_indices] = unique_join.inner_join();

  auto constexpr gold_size = 2;

  EXPECT_EQ(left_join_indices->size(), gold_size);
  EXPECT_EQ(right_join_indices->size(), gold_size);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
  auto left_result          = cudf::gather(build.view(), left_indices_col, oob_policy);
  auto right_result         = cudf::gather(probe.view(), right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  auto joined_table        = std::make_unique<cudf::table>(std::move(joined_cols));
  auto result_sort_order   = cudf::sorted_order(joined_table->view());
  auto sorted_joined_table = cudf::gather(joined_table->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  strcol_wrapper col_gold_1({"s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  strcol_wrapper col_gold_4({"s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{9, 9}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_joined_table);
}
