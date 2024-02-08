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

#include <cudf/join.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <limits>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct UniqueJoinTest : public cudf::test::BaseFixture {};

/*
// This function is a wrapper around cudf's join APIs that takes the gather map
// from join APIs and materializes the table that would be created by gathering
// from the joined tables. Join APIs originally returned tables like this, but
// they were modified in https://github.com/rapidsai/cudf/pull/7454. This
// helper function allows us to avoid rewriting all our tests in terms of
// gather maps.
template <std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>> (*join_impl)(
            cudf::table_view const& left_keys,
            cudf::table_view const& right_keys,
            cudf::null_equality compare_nulls,
            rmm::mr::device_memory_resource* mr),
          cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK>
std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    join_impl(left_selected, right_selected, compare_nulls, mr);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::inner_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::left_join, cudf::out_of_bounds_policy::NULLIFY>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& full_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& full_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::full_join, cudf::out_of_bounds_policy::NULLIFY>(
    full_input, right_input, full_on, right_on, compare_nulls);
}

struct UniqueJoinTest : public cudf::test::BaseFixture {
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> gather_maps_as_tables(
    cudf::column_view const& expected_left_map,
    cudf::column_view const& expected_right_map,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result)
  {
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data(),
                                          nullptr,
                                          0},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data(),
                                          nullptr,
                                          0}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    cudf::table_view gold({expected_left_map, expected_right_map});
    auto gold_sort_order = cudf::sorted_order(gold);
    auto sorted_gold     = cudf::gather(gold, *gold_sort_order);

    return std::pair(std::move(sorted_gold), std::move(sorted_result));
  }
};
*/

TEST_F(UniqueJoinTest, InnerJoinNoNulls)
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

  auto unique_join  = cudf::unique_hash_join<cudf::has_nested::YES>{build.view(), probe.view()};
  auto const result = unique_join.inner_join();

  auto constexpr gold_size = 2;
  EXPECT_EQ(result.first->size(), gold_size);

  /*
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  strcol_wrapper col_gold_1({"s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  strcol_wrapper col_gold_4({"s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{1, 0, 0}};
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
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  */
}

/*
TEST_F(UniqueJoinTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_5{{1, -1}, {1, 0}};
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
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}
*/
