/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <utility>

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr bool print_all{true};
constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

template <typename T>
struct TypedStructSearchTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedStructSearchTest, TestTypes);

namespace {
auto search_bounds(std::unique_ptr<cudf::column> const& t_col,
                   std::unique_ptr<cudf::column> const& values_col,
                   std::vector<cudf::order> const& column_orders        = {cudf::order::ASCENDING},
                   std::vector<cudf::null_order> const& null_precedence = {
                     cudf::null_order::BEFORE})
{
  auto const t            = cudf::table_view{std::vector<cudf::column_view>{t_col->view()}};
  auto const values       = cudf::table_view{std::vector<cudf::column_view>{values_col->view()}};
  auto result_lower_bound = cudf::lower_bound(t, values, column_orders, null_precedence);
  auto result_upper_bound = cudf::upper_bound(t, values, column_orders, null_precedence);
  return std::make_pair(std::move(result_lower_bound), std::move(result_upper_bound));
}
}  // namespace

// Test case when all input columns are empty
TYPED_TEST(TypedStructSearchTest, EmptyInputTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_t     = col_wrapper{};
  auto const structs_t = structs_col{{child_col_t}, std::vector<bool>{}}.release();

  auto child_col_values     = col_wrapper{};
  auto const structs_values = structs_col{{child_col_values}, std::vector<bool>{}}.release();

  auto const results  = search_bounds(structs_t, structs_values);
  auto const expected = int32s_col{};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results.first->view(), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results.second->view(), print_all);
}

TYPED_TEST(TypedStructSearchTest, TrivialInputTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_t     = col_wrapper{10, 20, 30, 40, 50};
  auto const structs_t = structs_col{{child_col_t}}.release();

  auto child_col_values1     = col_wrapper{0, 1, 2, 3, 4};
  auto const structs_values1 = structs_col{{child_col_values1}}.release();

  auto child_col_values2     = col_wrapper{100, 101, 102, 103, 104};
  auto const structs_values2 = structs_col{{child_col_values2}}.release();

  auto const results1  = search_bounds(structs_t, structs_values1);
  auto const expected1 = int32s_col{0, 0, 0, 0, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, results1.first->view(), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, results1.second->view(), print_all);

  auto const results2  = search_bounds(structs_t, structs_values2);
  auto const expected2 = int32s_col{5, 5, 5, 5, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, results2.first->view(), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, results2.second->view(), print_all);
}

TYPED_TEST(TypedStructSearchTest, SimpleInputWithNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_values =
    col_wrapper{{1, null, 70, XXX, 2, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_values = structs_col{
    {child_col_values}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  // Sorted asc, nulls first
  auto child_col_t1 =
    col_wrapper{{XXX, null, 0, 1, 2, 2, 2, 2, 3, 3, 4},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_t1 = structs_col{
    {child_col_t1}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 0;
    })}.release();

  auto results =
    search_bounds(structs_t1, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto expected_lower_bound = int32s_col{3, 1, 11, 0, 4, 11};
  auto expected_upper_bound = int32s_col{4, 2, 11, 1, 8, 11};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), print_all);

  // Sorted asc, nulls last
  auto child_col_t2 =
    col_wrapper{{0, 1, 2, 2, 2, 2, 3, 3, 4, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 9; })};
  auto const structs_t2 = structs_col{
    {child_col_t2}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 10;
    })}.release();
  results =
    search_bounds(structs_t2, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{1, 0, 10, 10, 2, 10};
  expected_upper_bound = int32s_col{1, 0, 10, 11, 6, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), print_all);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), print_all);

  // Sorted dsc, nulls first
  auto child_col_t3 =
    col_wrapper{{XXX, null, 4, 3, 3, 2, 2, 2, 2, 1, 0},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_t3 = structs_col{
    {child_col_t3}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 0;
    })}.release();
  results =
    search_bounds(structs_t2, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{1, 0, 10, 10, 2, 10};
  expected_upper_bound = int32s_col{1, 0, 10, 11, 6, 10};
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), print_all);
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), print_all);

  // Sorted dsc, nulls last
  auto child_col_t4 =
    col_wrapper{{4, 3, 3, 2, 2, 2, 2, 1, 0, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 9; })};
  auto const structs_t4 = structs_col{
    {child_col_t4}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 10;
    })}.release();
  results =
    search_bounds(structs_t2, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{1, 0, 10, 10, 2, 10};
  expected_upper_bound = int32s_col{1, 0, 10, 11, 6, 10};
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), print_all);
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), print_all);
}

TYPED_TEST(TypedStructSearchTest, ComplexStructTest)
{
  // Testing on struct<string, numeric, bool>.
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto names_column_values = strings_col{nullptr, "Bagel", "Lemonade", "Donut", "Butter"};
  auto ages_column_values =
    col_wrapper{{15, null, 10, 21, 17},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto is_human_col_values  = bools_col{false, false, false, false, false};
  auto const structs_values = structs_col{
    {names_column_values, ages_column_values, is_human_col_values},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 2;
    })}.release();

  auto names_column_t = strings_col{"Cherry", "Kiwi", "Lemon", "Newton", "Tomato", "Washington"};
  auto ages_column_t =
    col_wrapper{{5, 10, 15, 20, null, 30},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto is_human_col_t = bools_col{false, false, false, false, false, true};

  auto const structs_t = structs_col{
    {names_column_t, ages_column_t, is_human_col_t},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  auto results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto expected_lower_bound = int32s_col{3, 1, 11, 0, 4, 11};
  auto expected_upper_bound = int32s_col{4, 2, 11, 1, 8, 11};
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), print_all);
  //  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), print_all);
}
