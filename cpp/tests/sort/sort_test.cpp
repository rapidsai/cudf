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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <vector>

namespace cudf {
namespace test {
void run_sort_test(table_view input,
                   column_view expected_sorted_indices,
                   std::vector<order> column_order         = {},
                   std::vector<null_order> null_precedence = {})
{
  // Sorted table
  auto got_sorted_table      = sort(input, column_order, null_precedence);
  auto expected_sorted_table = gather(input, expected_sorted_indices);

  expect_tables_equal(expected_sorted_table->view(), got_sorted_table->view());

  // Sorted by key
  auto got_sort_by_key_table      = sort_by_key(input, input, column_order, null_precedence);
  auto expected_sort_by_key_table = gather(input, expected_sorted_indices);

  expect_tables_equal(expected_sort_by_key_table->view(), got_sort_by_key_table->view());
}

template <typename T>
struct Sort : public BaseFixture {
};

TYPED_TEST_CASE(Sort, NumericTypes);

TYPED_TEST(Sort, WithNullMax)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k", "d"}, {1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 1}};
  table_view input{{col1, col2, col3}};

  fixed_width_column_wrapper<int32_t> expected{{1, 0, 5, 3, 4, 2}};
  std::vector<order> column_order{order::ASCENDING, order::ASCENDING, order::DESCENDING};
  std::vector<null_order> null_precedence{null_order::AFTER, null_order::AFTER, null_order::AFTER};

  // Sorted order
  auto got = sorted_order(input, column_order, null_precedence);

  if (!std::is_same<T, bool>::value) {
    expect_columns_equal(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order, null_precedence);
  } else {
    // for bools only validate that the null element landed at the back, since
    // the rest of the values are equivalent and yields random sorted order.
    auto to_host = [](column_view const& col) {
      thrust::host_vector<int32_t> h_data(col.size());
      CUDA_TRY(cudaMemcpy(
        h_data.data(), col.data<int32_t>(), h_data.size() * sizeof(int32_t), cudaMemcpyDefault));
      return h_data;
    };
    thrust::host_vector<int32_t> h_exp = to_host(expected);
    thrust::host_vector<int32_t> h_got = to_host(got->view());
    EXPECT_EQ(h_exp[h_exp.size() - 1], h_got[h_got.size() - 1]);

    // Run test for sort and sort_by_key
    fixed_width_column_wrapper<int32_t> expected_for_bool{{0, 3, 5, 1, 4, 2}};
    run_sort_test(input, expected_for_bool, column_order, null_precedence);
  }
}

TYPED_TEST(Sort, WithNullMin)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {1, 1, 0, 1, 1}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"}, {1, 1, 0, 1, 1});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 1, 0, 1, 1}};
  table_view input{{col1, col2, col3}};

  fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
  std::vector<order> column_order{order::ASCENDING, order::ASCENDING, order::DESCENDING};

  auto got = sorted_order(input, column_order);

  if (!std::is_same<T, bool>::value) {
    cudf::test::expect_columns_equal(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // for bools only validate that the null element landed at the front, since
    // the rest of the values are equivalent and yields random sorted order.
    auto to_host = [](column_view const& col) {
      thrust::host_vector<int32_t> h_data(col.size());
      CUDA_TRY(cudaMemcpy(
        h_data.data(), col.data<int32_t>(), h_data.size() * sizeof(int32_t), cudaMemcpyDefault));
      return h_data;
    };
    thrust::host_vector<int32_t> h_exp = to_host(expected);
    thrust::host_vector<int32_t> h_got = to_host(got->view());
    EXPECT_EQ(h_exp.front(), h_got.front());

    // Run test for sort and sort_by_key
    fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 0, 3, 1, 4}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, WithMixedNullOrder)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {0, 0, 1, 1, 0}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"}, {0, 1, 0, 0, 1});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 0, 1, 0, 1}};
  table_view input{{col1, col2, col3}};

  fixed_width_column_wrapper<int32_t> expected{{2, 3, 0, 1, 4}};
  std::vector<order> column_order{order::ASCENDING, order::ASCENDING, order::ASCENDING};
  std::vector<null_order> null_precedence{null_order::AFTER, null_order::BEFORE, null_order::AFTER};

  auto got = sorted_order(input, column_order, null_precedence);

  if (!std::is_same<T, bool>::value) {
    cudf::test::expect_columns_equal(expected, got->view());
  } else {
    // for bools only validate that the null element landed at the front, since
    // the rest of the values are equivalent and yields random sorted order.
    auto to_host = [](column_view const& col) {
      thrust::host_vector<int32_t> h_data(col.size());
      CUDA_TRY(cudaMemcpy(
        h_data.data(), col.data<int32_t>(), h_data.size() * sizeof(int32_t), cudaMemcpyDefault));
      return h_data;
    };
    thrust::host_vector<int32_t> h_exp = to_host(expected);
    thrust::host_vector<int32_t> h_got = to_host(got->view());
    EXPECT_EQ(h_exp.front(), h_got.front());
  }

  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order, null_precedence);
}

TYPED_TEST(Sort, WithAllValid)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  table_view input{{col1, col2, col3}};

  fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
  std::vector<order> column_order{order::ASCENDING, order::ASCENDING, order::DESCENDING};

  auto got = sorted_order(input, column_order);

  // Skip validating bools order. Valid true bools are all
  // equivalent, and yield random order after thrust::sort
  if (!std::is_same<T, bool>::value) {
    cudf::test::expect_columns_equal(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // Run test for sort and sort_by_key
    fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 0, 3, 1, 4}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, Stable)
{
  using T = TypeParam;
  using R = int32_t;

  fixed_width_column_wrapper<T> col1({0, 1, 1, 0, 0, 1, 0, 1}, {0, 1, 1, 1, 1, 1, 1, 1});
  strings_column_wrapper col2({"2", "a", "b", "x", "k", "a", "x", "a"}, {1, 1, 1, 1, 0, 1, 1, 1});

  fixed_width_column_wrapper<R> expected{{4, 3, 6, 1, 5, 7, 2, 0}};

  auto got = stable_sorted_order(table_view({col1, col2}),
                                 {order::ASCENDING, order::ASCENDING},
                                 {null_order::AFTER, null_order::BEFORE});

  expect_columns_equal(expected, got->view());
}

TYPED_TEST(Sort, MisMatchInColumnOrderSize)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  table_view input{{col1, col2, col3}};

  std::vector<order> column_order{order::ASCENDING, order::DESCENDING};

  EXPECT_THROW(sorted_order(input, column_order), logic_error);
  EXPECT_THROW(stable_sorted_order(input, column_order), logic_error);
  EXPECT_THROW(sort(input, column_order), logic_error);
  EXPECT_THROW(sort_by_key(input, input, column_order), logic_error);
}

TYPED_TEST(Sort, MisMatchInNullPrecedenceSize)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  table_view input{{col1, col2, col3}};

  std::vector<order> column_order{order::ASCENDING, order::DESCENDING, order::DESCENDING};
  std::vector<null_order> null_precedence{null_order::AFTER, null_order::BEFORE};

  EXPECT_THROW(sorted_order(input, column_order, null_precedence), logic_error);
  EXPECT_THROW(stable_sorted_order(input, column_order, null_precedence), logic_error);
  EXPECT_THROW(sort(input, column_order, null_precedence), logic_error);
  EXPECT_THROW(sort_by_key(input, input, column_order, null_precedence), logic_error);
}

TYPED_TEST(Sort, ZeroSizedColumns)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{};
  table_view input{{col1}};

  fixed_width_column_wrapper<int32_t> expected{};
  std::vector<order> column_order{order::ASCENDING};

  auto got = sorted_order(input, column_order);

  expect_columns_equal(expected, got->view());

  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order);
}

struct SortByKey : public BaseFixture {
};

TEST_F(SortByKey, ValueKeysSizeMismatch)
{
  using T = int64_t;

  fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  table_view values{{col1, col2, col3}};

  fixed_width_column_wrapper<T> key_col{{5, 4, 3, 5}};
  table_view keys{{key_col}};

  EXPECT_THROW(sort_by_key(values, keys), logic_error);
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointSortedOrderGather)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ZERO  = decimalXX{0, scale_type{0}};
  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};

  auto const input_vec  = std::vector<decimalXX>{TWO, ONE, ZERO, FOUR, THREE};
  auto const index_vec  = std::vector<cudf::size_type>{2, 1, 0, 4, 3};
  auto const sorted_vec = std::vector<decimalXX>{ZERO, ONE, TWO, THREE, FOUR};

  auto const input_col  = wrapper<decimalXX>(input_vec.begin(), input_vec.end());
  auto const index_col  = wrapper<cudf::size_type>(index_vec.begin(), index_vec.end());
  auto const sorted_col = wrapper<decimalXX>(sorted_vec.begin(), sorted_vec.end());

  auto const sorted_table = cudf::table_view{{sorted_col}};
  auto const input_table  = cudf::table_view{{input_col}};

  auto const indices = cudf::sorted_order(input_table);
  auto const sorted  = cudf::gather(input_table, indices->view());

  cudf::test::expect_columns_equal(index_col, indices->view());
  cudf::test::expect_tables_equal(sorted_table, sorted->view());
}

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
