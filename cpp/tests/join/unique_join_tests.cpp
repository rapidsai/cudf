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

struct UniqueJoinTest : public cudf::test::BaseFixture {
  void compare_to_reference(
    cudf::table_view const& build_table,
    cudf::table_view const& probe_table,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result,
    cudf::table_view const& expected_table)
  {
    auto const& [build_join_indices, probe_join_indices] = result;

    auto build_indices_span = cudf::device_span<cudf::size_type const>{*build_join_indices};
    auto probe_indices_span = cudf::device_span<cudf::size_type const>{*probe_join_indices};

    auto build_indices_col = cudf::column_view{build_indices_span};
    auto probe_indices_col = cudf::column_view{probe_indices_span};

    auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    auto build_result         = cudf::gather(build_table, build_indices_col, oob_policy);
    auto probe_result         = cudf::gather(probe_table, probe_indices_col, oob_policy);

    auto joined_cols = build_result->release();
    auto right_cols  = probe_result->release();
    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    auto joined_table        = std::make_unique<cudf::table>(std::move(joined_cols));
    auto result_sort_order   = cudf::sorted_order(joined_table->view());
    auto sorted_joined_table = cudf::gather(joined_table->view(), *result_sort_order);

    auto expected_sort_order = cudf::sorted_order(expected_table);
    auto sorted_expected     = cudf::gather(expected_table, *expected_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_joined_table);
  }
};

TEST_F(UniqueJoinTest, IntegerInnerJoin)
{
  auto constexpr size = 2024;

  auto const init = cudf::numeric_scalar<int32_t>{0};

  auto build = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{1});
  auto probe = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{2});

  auto build_table = cudf::table_view{{build->view()}};
  auto probe_table = cudf::table_view{{probe->view()}};

  auto unique_join =
    cudf::unique_hash_join<cudf::has_nested::NO>{build_table, probe_table, cudf::nullable_join::NO};

  auto result = unique_join.inner_join();

  auto constexpr gold_size = size / 2;
  auto gold                = cudf::sequence(gold_size, init, cudf::numeric_scalar<int32_t>{2});
  this->compare_to_reference(build_table, probe_table, result, cudf::table_view{{gold->view()}});
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
  auto result      = unique_join.inner_join();

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

  this->compare_to_reference(build.view(), probe.view(), result, gold.view());
}
