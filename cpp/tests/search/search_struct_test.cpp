/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>

using namespace cudf::test::iterators;

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

template <typename T>
struct TypedStructSearchTest : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(TypedStructSearchTest, TestTypes);

namespace {
auto search_bounds(cudf::column_view const& t_col_view,
                   std::unique_ptr<cudf::column> const& values_col,
                   std::vector<cudf::order> const& column_orders        = {cudf::order::ASCENDING},
                   std::vector<cudf::null_order> const& null_precedence = {
                     cudf::null_order::BEFORE})
{
  auto const t            = cudf::table_view{std::vector<cudf::column_view>{t_col_view}};
  auto const values       = cudf::table_view{std::vector<cudf::column_view>{values_col->view()}};
  auto result_lower_bound = cudf::lower_bound(t, values, column_orders, null_precedence);
  auto result_upper_bound = cudf::upper_bound(t, values, column_orders, null_precedence);
  return std::pair(std::move(result_lower_bound), std::move(result_upper_bound));
}

auto search_bounds(std::unique_ptr<cudf::column> const& t_col,
                   std::unique_ptr<cudf::column> const& values_col,
                   std::vector<cudf::order> const& column_orders        = {cudf::order::ASCENDING},
                   std::vector<cudf::null_order> const& null_precedence = {
                     cudf::null_order::BEFORE})
{
  return search_bounds(t_col->view(), values_col, column_orders, null_precedence);
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results.second->view(), verbosity);
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, results1.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, results1.second->view(), verbosity);

  auto const results2  = search_bounds(structs_t, structs_values2);
  auto const expected2 = int32s_col{5, 5, 5, 5, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, results2.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, results2.second->view(), verbosity);
}

TYPED_TEST(TypedStructSearchTest, SlicedColumnInputTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_values     = col_wrapper{0, 1, 2, 3, 4, 5};
  auto const structs_values = structs_col{child_col_values}.release();

  auto child_col_t              = col_wrapper{0, 1, 2, 2, 2, 2, 3, 3, 4, 4};
  auto const structs_t_original = structs_col{child_col_t}.release();

  auto structs_t = cudf::slice(structs_t_original->view(), {0, 10})[0];  // the entire column t
  auto results   = search_bounds(structs_t, structs_values);
  auto expected_lower_bound = int32s_col{0, 1, 2, 6, 8, 10};
  auto expected_upper_bound = int32s_col{1, 2, 6, 8, 10, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  structs_t            = cudf::slice(structs_t_original->view(), {0, 5})[0];
  results              = search_bounds(structs_t, structs_values);
  expected_lower_bound = int32s_col{0, 1, 2, 5, 5, 5};
  expected_upper_bound = int32s_col{1, 2, 5, 5, 5, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  structs_t            = cudf::slice(structs_t_original->view(), {5, 10})[0];
  results              = search_bounds(structs_t, structs_values);
  expected_lower_bound = int32s_col{0, 0, 0, 1, 3, 5};
  expected_upper_bound = int32s_col{0, 0, 1, 3, 5, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);
}

TYPED_TEST(TypedStructSearchTest, SimpleInputWithNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_values     = col_wrapper{{1, null, 70, XXX, 2, 100}, null_at(1)};
  auto const structs_values = structs_col{{child_col_values}, null_at(3)}.release();

  // Sorted asc, nulls first
  auto child_col_t = col_wrapper{{XXX, null, 0, 1, 2, 2, 2, 2, 3, 3, 4}, null_at(1)};
  auto structs_t   = structs_col{{child_col_t}, null_at(0)}.release();

  auto results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto expected_lower_bound = int32s_col{3, 1, 11, 0, 4, 11};
  auto expected_upper_bound = int32s_col{4, 2, 11, 1, 8, 11};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted asc, nulls last
  child_col_t = col_wrapper{{0, 1, 2, 2, 2, 2, 3, 3, 4, null, XXX}, null_at(9)};
  structs_t   = structs_col{{child_col_t}, null_at(10)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{1, 9, 9, 10, 2, 9};
  expected_upper_bound = int32s_col{2, 10, 9, 11, 6, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, nulls first
  child_col_t = col_wrapper{{XXX, null, 4, 3, 3, 2, 2, 2, 2, 1, 0}, null_at(1)};
  structs_t   = structs_col{{child_col_t}, null_at(0)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
  expected_lower_bound = int32s_col{9, 11, 0, 11, 5, 0};
  expected_upper_bound = int32s_col{10, 11, 0, 11, 9, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, nulls last
  child_col_t = col_wrapper{{4, 3, 3, 2, 2, 2, 2, 1, 0, null, XXX}, null_at(9)};
  structs_t   = structs_col{{child_col_t}, null_at(10)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{7, 0, 0, 0, 3, 0};
  expected_upper_bound = int32s_col{8, 0, 0, 0, 7, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);
}

TYPED_TEST(TypedStructSearchTest, SimpleInputWithValuesHavingNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_values     = col_wrapper{{1, null, 70, XXX, 2, 100}, null_at(1)};
  auto const structs_values = structs_col{{child_col_values}, null_at(3)}.release();

  // Sorted asc, search nulls first
  auto child_col_t = col_wrapper{0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 4};
  auto structs_t   = structs_col{{child_col_t}}.release();

  auto results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto expected_lower_bound = int32s_col{3, 0, 11, 0, 4, 11};
  auto expected_upper_bound = int32s_col{4, 0, 11, 0, 8, 11};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted asc, search nulls last
  results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{3, 11, 11, 11, 4, 11};
  expected_upper_bound = int32s_col{4, 11, 11, 11, 8, 11};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, search nulls first
  child_col_t = col_wrapper{4, 3, 3, 2, 2, 2, 2, 1, 0, 0, 0};
  structs_t   = structs_col{{child_col_t}}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
  expected_lower_bound = int32s_col{7, 11, 0, 11, 3, 0};
  expected_upper_bound = int32s_col{8, 11, 0, 11, 7, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, search nulls last
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{7, 0, 0, 0, 3, 0};
  expected_upper_bound = int32s_col{8, 0, 0, 0, 7, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);
}

TYPED_TEST(TypedStructSearchTest, SimpleInputWithTargetHavingNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_values     = col_wrapper{1, 0, 70, 0, 2, 100};
  auto const structs_values = structs_col{{child_col_values}}.release();

  // Sorted asc, nulls first
  auto child_col_t = col_wrapper{{XXX, null, 0, 1, 2, 2, 2, 2, 3, 3, 4}, null_at(1)};
  auto structs_t   = structs_col{{child_col_t}, null_at(0)}.release();

  auto results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  auto expected_lower_bound = int32s_col{3, 2, 11, 2, 4, 11};
  auto expected_upper_bound = int32s_col{4, 3, 11, 3, 8, 11};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted asc, nulls last
  child_col_t = col_wrapper{{0, 1, 2, 2, 2, 2, 3, 3, 4, null, XXX}, null_at(9)};
  structs_t   = structs_col{{child_col_t}, null_at(10)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{1, 0, 9, 0, 2, 9};
  expected_upper_bound = int32s_col{2, 1, 9, 1, 6, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, nulls first
  child_col_t = col_wrapper{{XXX, null, 4, 3, 3, 2, 2, 2, 2, 1, 0}, null_at(1)};
  structs_t   = structs_col{{child_col_t}, null_at(0)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
  expected_lower_bound = int32s_col{9, 10, 0, 10, 5, 0};
  expected_upper_bound = int32s_col{10, 11, 0, 11, 9, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);

  // Sorted dsc, nulls last
  child_col_t = col_wrapper{{4, 3, 3, 2, 2, 2, 2, 1, 0, null, XXX}, null_at(9)};
  structs_t   = structs_col{{child_col_t}, null_at(10)}.release();
  results =
    search_bounds(structs_t, structs_values, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
  expected_lower_bound = int32s_col{7, 8, 0, 8, 3, 0};
  expected_upper_bound = int32s_col{8, 11, 0, 11, 7, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);
}

TYPED_TEST(TypedStructSearchTest, OneColumnHasNullMaskButNoNullElementTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col1         = col_wrapper{1, 20, 30};
  auto const structs_col1 = structs_col{{child_col1}}.release();

  auto child_col2         = col_wrapper{0, 10, 10};
  auto const structs_col2 = structs_col{child_col2}.release();

  // structs_col3 (and its child column) will have a null mask but no null element
  auto child_col3         = col_wrapper{{0, 10, 10}, no_nulls()};
  auto const structs_col3 = structs_col{{child_col3}, no_nulls()}.release();

  // Search struct elements of structs_col2 and structs_col3 in the column structs_col1
  {
    auto const results1             = search_bounds(structs_col1, structs_col2);
    auto const results2             = search_bounds(structs_col1, structs_col3);
    auto const expected_lower_bound = int32s_col{0, 1, 1};
    auto const expected_upper_bound = int32s_col{0, 1, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results1.first->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results2.first->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results1.second->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results2.second->view(), verbosity);
  }

  // Search struct elements of structs_col1 in the columns structs_col2 and structs_col3
  {
    auto const results1             = search_bounds(structs_col2, structs_col1);
    auto const results2             = search_bounds(structs_col3, structs_col1);
    auto const expected_lower_bound = int32s_col{1, 3, 3};
    auto const expected_upper_bound = int32s_col{1, 3, 3};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results1.first->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results2.first->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results1.second->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results2.second->view(), verbosity);
  }
}

TYPED_TEST(TypedStructSearchTest, ComplexStructTest)
{
  // Testing on struct<string, numeric, bool>.
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto names_column_t =
    strings_col{"Cherry", "Kiwi", "Lemon", "Newton", "Tomato", /*NULL*/ "Washington"};
  auto ages_column_t  = col_wrapper{{5, 10, 15, 20, null, XXX}, null_at(4)};
  auto is_human_col_t = bools_col{false, false, false, false, false, /*NULL*/ true};

  auto const structs_t =
    structs_col{{names_column_t, ages_column_t, is_human_col_t}, null_at(5)}.release();

  auto names_column_values = strings_col{"Bagel", "Tomato", "Lemonade", /*NULL*/ "Donut", "Butter"};
  auto ages_column_values  = col_wrapper{{10, null, 15, XXX, 17}, null_at(1)};
  auto is_human_col_values = bools_col{false, false, true, /*NULL*/ true, true};
  auto const structs_values =
    structs_col{{names_column_values, ages_column_values, is_human_col_values}, null_at(3)}
      .release();

  auto const results =
    search_bounds(structs_t, structs_values, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  auto const expected_lower_bound = int32s_col{0, 4, 3, 5, 0};
  auto const expected_upper_bound = int32s_col{0, 5, 3, 6, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, results.first->view(), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, results.second->view(), verbosity);
}

template <typename T>
struct TypedScalarStructContainTest : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(TypedScalarStructContainTest, TestTypes);

TYPED_TEST(TypedScalarStructContainTest, EmptyInputTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const col = [] {
    auto child = col_wrapper{};
    return structs_col{{child}};
  }();

  auto const val = [] {
    auto child = col_wrapper{1};
    return cudf::struct_scalar(std::vector<cudf::column_view>{child});
  }();

  EXPECT_EQ(false, cudf::contains(col, val));
}

TYPED_TEST(TypedScalarStructContainTest, TrivialInputTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const col = [] {
    auto child1 = col_wrapper{1, 2, 3};
    auto child2 = col_wrapper{4, 5, 6};
    auto child3 = strings_col{"x", "y", "z"};
    return structs_col{{child1, child2, child3}};
  }();

  auto const val1 = [] {
    auto child1 = col_wrapper{1};
    auto child2 = col_wrapper{4};
    auto child3 = strings_col{"x"};
    return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
  }();
  auto const val2 = [] {
    auto child1 = col_wrapper{1};
    auto child2 = col_wrapper{4};
    auto child3 = strings_col{"a"};
    return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
  }();

  EXPECT_EQ(true, cudf::contains(col, val1));
  EXPECT_EQ(false, cudf::contains(col, val2));
}

TYPED_TEST(TypedScalarStructContainTest, SlicedColumnInputTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  constexpr int32_t dont_care{0};

  auto const col_original = [] {
    auto child1 = col_wrapper{dont_care, dont_care, 1, 2, 3, dont_care};
    auto child2 = col_wrapper{dont_care, dont_care, 4, 5, 6, dont_care};
    auto child3 = strings_col{"dont_care", "dont_care", "x", "y", "z", "dont_care"};
    return structs_col{{child1, child2, child3}};
  }();
  auto const col = cudf::slice(col_original, {2, 5})[0];

  auto const val1 = [] {
    auto child1 = col_wrapper{1};
    auto child2 = col_wrapper{4};
    auto child3 = strings_col{"x"};
    return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
  }();
  auto const val2 = [] {
    auto child1 = col_wrapper{dont_care};
    auto child2 = col_wrapper{dont_care};
    auto child3 = strings_col{"dont_care"};
    return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
  }();

  EXPECT_EQ(true, cudf::contains(col, val1));
  EXPECT_EQ(false, cudf::contains(col, val2));
}

TYPED_TEST(TypedScalarStructContainTest, SimpleInputWithNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  constexpr int32_t null{0};

  // Test with nulls at the top level.
  {
    auto const col = [] {
      auto child1 = col_wrapper{1, null, 3};
      auto child2 = col_wrapper{4, null, 6};
      auto child3 = strings_col{"x", "" /*NULL*/, "z"};
      return structs_col{{child1, child2, child3}, null_at(1)};
    }();

    auto const val1 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{"x"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();
    auto const val2 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{"a"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();

    EXPECT_EQ(true, cudf::contains(col, val1));
    EXPECT_EQ(false, cudf::contains(col, val2));
  }

  // Test with nulls at the children level.
  {
    auto const col = [] {
      auto child1 = col_wrapper{{1, null, 3}, null_at(1)};
      auto child2 = col_wrapper{{4, null, 6}, null_at(1)};
      auto child3 = strings_col{{"" /*NULL*/, "y", "z"}, null_at(0)};
      return structs_col{{child1, child2, child3}};
    }();

    auto const val1 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{{"" /*NULL*/}, null_at(0)};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();
    auto const val2 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{""};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();

    EXPECT_EQ(true, cudf::contains(col, val1));
    EXPECT_EQ(false, cudf::contains(col, val2));
  }

  // Test with nulls in the input scalar.
  {
    auto const col = [] {
      auto child1 = col_wrapper{1, 2, 3};
      auto child2 = col_wrapper{4, 5, 6};
      auto child3 = strings_col{"x", "y", "z"};
      return structs_col{{child1, child2, child3}};
    }();

    auto const val1 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{"x"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();
    auto const val2 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{{"" /*NULL*/}, null_at(0)};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();

    EXPECT_EQ(true, cudf::contains(col, val1));
    EXPECT_EQ(false, cudf::contains(col, val2));
  }
}

TYPED_TEST(TypedScalarStructContainTest, SlicedInputWithNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  constexpr int32_t dont_care{0};
  constexpr int32_t null{0};

  // Test with nulls at the top level.
  {
    auto const col_original = [] {
      auto child1 = col_wrapper{dont_care, dont_care, 1, null, 3, dont_care};
      auto child2 = col_wrapper{dont_care, dont_care, 4, null, 6, dont_care};
      auto child3 = strings_col{"dont_care", "dont_care", "x", "" /*NULL*/, "z", "dont_care"};
      return structs_col{{child1, child2, child3}, null_at(3)};
    }();
    auto const col = cudf::slice(col_original, {2, 5})[0];

    auto const val1 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{"x"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();
    auto const val2 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{"a"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();

    EXPECT_EQ(true, cudf::contains(col, val1));
    EXPECT_EQ(false, cudf::contains(col, val2));
  }

  // Test with nulls at the children level.
  {
    auto const col_original = [] {
      auto child1 =
        col_wrapper{{dont_care, dont_care /*also NULL*/, 1, null, 3, dont_care}, null_at(3)};
      auto child2 =
        col_wrapper{{dont_care, dont_care /*also NULL*/, 4, null, 6, dont_care}, null_at(3)};
      auto child3 = strings_col{
        {"dont_care", "dont_care" /*also NULL*/, "" /*NULL*/, "y", "z", "dont_care"}, null_at(2)};
      return structs_col{{child1, child2, child3}, null_at(1)};
    }();
    auto const col = cudf::slice(col_original, {2, 5})[0];

    auto const val1 = [] {
      auto child1 = col_wrapper{1};
      auto child2 = col_wrapper{4};
      auto child3 = strings_col{{"x"}, null_at(0)};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();
    auto const val2 = [] {
      auto child1 = col_wrapper{dont_care};
      auto child2 = col_wrapper{dont_care};
      auto child3 = strings_col{"dont_care"};
      return cudf::struct_scalar(std::vector<cudf::column_view>{child1, child2, child3});
    }();

    EXPECT_EQ(true, cudf::contains(col, val1));
    EXPECT_EQ(false, cudf::contains(col, val2));
  }
}
