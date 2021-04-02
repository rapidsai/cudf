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

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

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
void test_search(std::unique_ptr<cudf::column> const& t_col,
                 std::unique_ptr<cudf::column> const& values_col,
                 int32s_col const& expected_lower_bound,
                 int32s_col const& expected_upper_bound,
                 std::vector<cudf::order> const& column_orders        = {cudf::order::ASCENDING},
                 std::vector<cudf::null_order> const& null_precedence = {cudf::null_order::BEFORE})
{
  auto const t      = cudf::table_view{std::vector<cudf::column_view>{t_col->view()}};
  auto const values = cudf::table_view{std::vector<cudf::column_view>{values_col->view()}};

  auto const result_lower_bound = cudf::lower_bound(t, values, column_orders, null_precedence);
  auto const result_upper_bound = cudf::upper_bound(t, values, column_orders, null_precedence);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, result_lower_bound->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, result_upper_bound->view());
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

  auto const expected = int32s_col{};
  test_search(structs_t, structs_values, expected, expected);
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

  auto const expected1 = int32s_col{0, 0, 0, 0, 0};
  auto const expected2 = int32s_col{5, 5, 5, 5, 5};
  test_search(structs_t, structs_values1, expected1, expected1);
  test_search(structs_t, structs_values2, expected2, expected2);
}

TYPED_TEST(TypedStructSearchTest, SimpleInputWithNullsTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto child_col_t =
    col_wrapper{{0, 1, 2, 3, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto const structs_t = structs_col{
    {child_col_t}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  // Target data
  auto child_col_values =
    col_wrapper{{50, null, 70, XXX, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_values = structs_col{
    {child_col_values}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  // Expected data
  auto child_col_expected1 =
    col_wrapper{{1, null, 70, XXX, 0, 2},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_expected1 = structs_col{
    {child_col_expected1}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();
  auto const scatter_map1 = int32s_col{-2, 0, 5}.release();
  test_search(structs_t, structs_values, structs_expected1, scatter_map1);

  // Expected data
  auto child_col_expected2 =
    col_wrapper{{1, null, 70, 3, 0, 2},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_expected2 = structs_col{
    {child_col_expected2}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return true;
    })}.release();
  auto const scatter_map2 = int32s_col{-2, 0, 5, 3}.release();
  test_search(structs_t, structs_values, structs_expected2, scatter_map2);
}
#if 0


TYPED_TEST(TypedStructSearchTest, ComplexDataScatterTest)
{
  // Testing scatter() on struct<string, numeric, bool>.
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto names_column_t =
    strings_col{{"Newton", "Washington", "Cherry", "Kiwi", "Lemon", "Tomato"},
                cudf::detail::make_counting_transform_iterator(0, [](auto) { return true; })};
  auto ages_column_t =
    col_wrapper{{5, 10, 15, 20, 25, 30},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto is_human_col_t =
    bools_col{{true, true, false, false, false, false},
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; })};

  // Target data
  auto names_column_values =
    strings_col{{"String 0", "String 1", "String 2", "String 3", "String 4", "String 5"},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; })};
  auto ages_column_values =
    col_wrapper{{50, 60, 70, 80, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto is_human_col_values =
    bools_col{{true, true, true, true, true, true},
              cudf::detail::make_counting_transform_iterator(0, [](auto) { return true; })};

  // Expected data
  auto names_column_expected =
    strings_col{{"String 0", "Lemon", "Kiwi", "Cherry", "Washington", "Newton"},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; })};
  auto ages_column_expected =
    col_wrapper{{50, 25, 20, 15, 10, 5},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto is_human_col_expected =
    bools_col{{true, false, false, false, true, true},
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2; })};

  auto const structs_t = structs_col{
    {names_column_t, ages_column_t, is_human_col_t},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();
  auto const structs_values = structs_col{
    {names_column_values, ages_column_values, is_human_col_values},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 2;
    })}.release();
  auto const structs_expected = structs_col{
    {names_column_expected, ages_column_expected, is_human_col_expected},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return true;
    })}.release();

  // The first element of the target is not overwritten
  auto const scatter_map = int32s_col{-1, 4, 3, 2, 1}.release();
  test_search(structs_t, structs_values, structs_expected, scatter_map);
}

TYPED_TEST(TypedStructSearchTest, ScatterStructOfListsTest)
{
  // Testing gather() on struct<list<numeric>>
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto lists_col_t =
    lists_col{{{5}, {10, 15}, {20, 25, 30}, {35, 40, 45, 50}, {55, 60, 65}, {70, 75}, {80}, {}, {}},
              // Valid for elements 0, 3, 6,...
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return !(i % 3); })};
  auto const structs_t = structs_col{{lists_col_t}}.release();

  // Target data
  auto lists_col_values =
    lists_col{{{1}, {2, 3}, {4, 5, 6}, {7, 8}, {9}, {10, 11, 12, 13}, {}, {14}, {15, 16}},
              // Valid for elements 1, 3, 5, 7,...
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })};
  auto const structs_values = structs_col{{lists_col_values}}.release();

  // Expected data
  auto const validity_expected = std::vector<bool>{0, 1, 1, 0, 0, 1, 1, 0, 0};
  auto lists_col_expected      = lists_col{
    {{1}, {2, 3}, {80}, {70, 75}, {55, 60, 65}, {35, 40, 45, 50}, {5}, {10, 15}, {20, 25, 30}},
    validity_expected.begin()};
  auto const structs_expected = structs_col{{lists_col_expected}}.release();

  // The first 2 elements of the target is not overwritten
  auto const scatter_map = int32s_col{-3, -2, -1, 5, 4, 3, 2}.release();
  test_search(structs_t, structs_values, structs_expected, scatter_map);
}

struct StructSearchTest : public cudf::test::BaseFixture {
};

using cudf::numeric_scalar;
using cudf::size_type;
using cudf::string_scalar;
using cudf::test::fixed_width_column_wrapper;

TEST_F(StructSearchTest, search_dictionary)
{
  cudf::test::dictionary_column_wrapper<std::string> input(
    {"", "", "10", "10", "20", "20", "30", "40"}, {0, 0, 1, 1, 1, 1, 1, 1});
  cudf::test::dictionary_column_wrapper<std::string> values(
    {"", "08", "10", "11", "30", "32", "90"}, {0, 1, 1, 1, 1, 1, 1});

  auto result = cudf::upper_bound({cudf::table_view{{input}}},
                                  {cudf::table_view{{values}}},
                                  {cudf::order::ASCENDING},
                                  {cudf::null_order::BEFORE});
  fixed_width_column_wrapper<size_type> expect_upper{2, 2, 4, 4, 7, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_upper);

  result = cudf::lower_bound({cudf::table_view{{input}}},
                             {cudf::table_view{{values}}},
                             {cudf::order::ASCENDING},
                             {cudf::null_order::BEFORE});
  fixed_width_column_wrapper<size_type> expect_lower{0, 2, 2, 4, 6, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_lower);
}

TEST_F(StructSearchTest, search_table_dictionary)
{
  fixed_width_column_wrapper<int32_t> column_0{{10, 10, 20, 20, 20, 20, 20, 20, 20, 50, 30},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};
  fixed_width_column_wrapper<float> column_1{{5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  cudf::test::dictionary_column_wrapper<int16_t> column_2{
    {90, 95, 77, 78, 79, 76, 61, 62, 63, 41, 50}, {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1}};
  cudf::table_view input({column_0, column_1, column_2});

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {1, 0, 1}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};
  cudf::test::dictionary_column_wrapper<int16_t> values_2{{95, 50, 77}, {1, 1, 0}};
  cudf::table_view values({values_0, values_1, values_2});

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  auto result = cudf::lower_bound(input, values, order_flags, null_order_flags);
  fixed_width_column_wrapper<size_type> expect_lower{1, 10, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_lower);

  result = cudf::upper_bound(input, values, order_flags, null_order_flags);
  fixed_width_column_wrapper<size_type> expect_upper{2, 11, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_upper);
}

TEST_F(StructSearchTest, contains_dictionary)
{
  cudf::test::dictionary_column_wrapper<std::string> column(
    {"00", "00", "17", "17", "23", "23", "29"});
  EXPECT_TRUE(cudf::contains(column, string_scalar{"23"}));
  EXPECT_FALSE(cudf::contains(column, string_scalar{"28"}));

  cudf::test::dictionary_column_wrapper<std::string> needles({"00", "17", "23", "27"});
  fixed_width_column_wrapper<bool> expect{1, 1, 1, 1, 1, 1, 0};
  auto result = cudf::contains(column, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}
#endif
