/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "row_operator_tests_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>

template <typename T>
struct TypedTableViewTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(TypedTableViewTest, NumericTypesNotBool);

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::experimental::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream);

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorTwoTables)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 4, 3}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const lhs          = cudf::table_view{{col1}};
  auto const rhs          = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 1, 0, 1}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got = two_table_comparison(
    lhs,
    rhs,
    column_order,
    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorSameTable)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const input_table  = cudf::table_view{{col1}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got =
    self_comparison(input_table,
                    column_order,
                    cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got =
    self_comparison(input_table,
                    column_order,
                    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

TYPED_TEST(TypedTableViewTest, TestSortSameTableFromTwoTables)
{
  using data_col   = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const col1      = data_col{5, 2, 7, 1, 3};
  auto const col2      = data_col{};  // empty
  auto const lhs       = cudf::table_view{{col1}};
  auto const empty_rhs = cudf::table_view{{col2}};

  auto const stream    = cudf::get_default_stream();
  auto const test_sort = [stream](auto const& preprocessed,
                                  auto const& input,
                                  auto const& comparator,
                                  auto const& expected) {
    auto const order = sorted_order(
      preprocessed, input.num_rows(), cudf::detail::has_nested_columns(input), comparator, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, order->view());
  };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{3, 1, 4, 0, 2};
    test_sort(preprocessed_lhs,
              lhs,
              cudf::experimental::row::lexicographic::physical_element_comparator{},
              expected_lhs);
    test_sort(preprocessed_lhs,
              lhs,
              cudf::experimental::row::lexicographic::sorting_physical_element_comparator{},
              expected_lhs);

    auto const expected_empty_rhs = int32s_col{};
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::experimental::row::lexicographic::physical_element_comparator{},
              expected_empty_rhs);
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::experimental::row::lexicographic::sorting_physical_element_comparator{},
              expected_empty_rhs);
  };

  // Generate preprocessed data for both lhs and lhs at the same time.
  // Switching order of lhs and rhs tables then sorting them using their preprocessed data should
  // produce exactly the same result.
  {
    auto const [preprocessed_lhs, preprocessed_empty_rhs] =
      cudf::experimental::row::lexicographic::preprocessed_table::create(
        lhs, empty_rhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
  {
    auto const [preprocessed_empty_rhs, preprocessed_lhs] =
      cudf::experimental::row::lexicographic::preprocessed_table::create(
        empty_rhs, lhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
}

TYPED_TEST(TypedTableViewTest, TestSortSameTableFromTwoTablesWithListsOfStructs)
{
  using data_col    = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const col1 = [] {
    auto const get_structs = [] {
      auto child0 = data_col{0, 3, 0, 2};
      auto child1 = strings_col{"a", "c", "a", "b"};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(
      2, int32s_col{0, 2, 4}.release(), get_structs().release(), 0, {});
  }();
  auto const col2 = [] {
    auto const get_structs = [] {
      auto child0 = data_col{};
      auto child1 = strings_col{};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(0, int32s_col{}.release(), get_structs().release(), 0, {});
  }();

  auto const column_order = std::vector{cudf::order::ASCENDING};
  auto const lhs          = cudf::table_view{{*col1}};
  auto const empty_rhs    = cudf::table_view{{*col2}};

  auto const stream    = cudf::get_default_stream();
  auto const test_sort = [stream](auto const& preprocessed,
                                  auto const& input,
                                  auto const& comparator,
                                  auto const& expected) {
    auto const order = sorted_order(
      preprocessed, input.num_rows(), cudf::detail::has_nested_columns(input), comparator, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, order->view());
  };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{1, 0};
    test_sort(preprocessed_lhs,
              lhs,
              cudf::experimental::row::lexicographic::sorting_physical_element_comparator{},
              expected_lhs);

    auto const expected_empty_rhs = int32s_col{};
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::experimental::row::lexicographic::sorting_physical_element_comparator{},
              expected_empty_rhs);

    EXPECT_THROW(test_sort(preprocessed_lhs,
                           lhs,
                           cudf::experimental::row::lexicographic::physical_element_comparator{},
                           expected_lhs),
                 cudf::logic_error);
    EXPECT_THROW(test_sort(preprocessed_empty_rhs,
                           empty_rhs,
                           cudf::experimental::row::lexicographic::physical_element_comparator{},
                           expected_empty_rhs),
                 cudf::logic_error);
  };

  // Generate preprocessed data for both lhs and lhs at the same time.
  // Switching order of lhs and rhs tables then sorting them using their preprocessed data should
  // produce exactly the same result.
  {
    auto const [preprocessed_lhs, preprocessed_empty_rhs] =
      cudf::experimental::row::lexicographic::preprocessed_table::create(
        lhs, empty_rhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
  {
    auto const [preprocessed_empty_rhs, preprocessed_lhs] =
      cudf::experimental::row::lexicographic::preprocessed_table::create(
        empty_rhs, lhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
}

template <typename T>
struct NaNTableViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NaNTableViewTest, cudf::test::FloatingPointTypes);

TYPED_TEST(NaNTableViewTest, TestLexicographicalComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 1, 0, 0}};
  auto const sorting_got      = two_table_comparison(
    lhs,
    rhs,
    column_order,
    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorting_expected, sorting_got->view());
}

TYPED_TEST(NaNTableViewTest, TestEqualityComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 1}};
  auto const got      = two_table_equality(
    lhs, rhs, column_order, cudf::experimental::row::equality::physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const nan_equal_expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 0, 0, 1}};
  auto const nan_equal_got =
    two_table_equality(lhs,
                       rhs,
                       column_order,
                       cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(nan_equal_expected, nan_equal_got->view());
}
