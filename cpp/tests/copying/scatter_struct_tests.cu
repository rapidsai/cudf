/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

template <typename T>
struct TypedStructScatterTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypes,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedStructScatterTest, TestTypes);

namespace {
void test_scatter(std::unique_ptr<cudf::column> const& structs_src,
                  std::unique_ptr<cudf::column> const& structs_tgt,
                  std::unique_ptr<cudf::column> const& structs_expected,
                  std::unique_ptr<cudf::column> const& scatter_map)
{
  auto const source = cudf::table_view{std::vector<cudf::column_view>{structs_src->view()}};
  auto const target = cudf::table_view{std::vector<cudf::column_view>{structs_tgt->view()}};
  auto const result = cudf::scatter(source, scatter_map->view(), target);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(structs_expected->view(), result->get_column(0));
}
}  // namespace

// Test case when all input columns are empty
TYPED_TEST(TypedStructScatterTest, EmptyInputTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{};
  auto const structs_src = structs_col{{child_col_src}, std::vector<bool>{}}.release();

  auto child_col_tgt     = col_wrapper{};
  auto const structs_tgt = structs_col{{child_col_tgt}, std::vector<bool>{}}.release();

  auto const scatter_map = int32s_col{}.release();
  test_scatter(structs_src, structs_tgt, structs_src, scatter_map);
  test_scatter(structs_src, structs_tgt, structs_tgt, scatter_map);
}

// Test case when only the scatter map is empty
TYPED_TEST(TypedStructScatterTest, EmptyScatterMapTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src =
    col_wrapper{{0, 1, 2, 3, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto const structs_src = structs_col{
    {child_col_src}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  auto child_col_tgt =
    col_wrapper{{50, null, 70, XXX, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_tgt = structs_col{
    {child_col_tgt}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  auto const scatter_map = int32s_col{}.release();
  test_scatter(structs_src, structs_tgt, structs_tgt, scatter_map);
}

TYPED_TEST(TypedStructScatterTest, ScatterAsCopyTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src =
    col_wrapper{{0, 1, 2, 3, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto const structs_src = structs_col{
    {child_col_src}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  auto child_col_tgt =
    col_wrapper{{50, null, 70, XXX, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_tgt = structs_col{
    {child_col_tgt}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  // Scatter as copy: the target should be the same as source
  auto const scatter_map = int32s_col{0, 1, 2, 3, 4, 5}.release();
  test_scatter(structs_src, structs_tgt, structs_src, scatter_map);
}

TYPED_TEST(TypedStructScatterTest, ScatterAsLeftShiftTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src =
    col_wrapper{{0, 1, 2, 3, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto const structs_src = structs_col{
    {child_col_src}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  auto child_col_tgt =
    col_wrapper{{50, null, 70, XXX, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_tgt = structs_col{
    {child_col_tgt}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  auto child_col_expected =
    col_wrapper{{2, 3, null, XXX, 0, 1},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2; })};
  auto structs_expected = structs_col{
    {child_col_expected}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 3;
    })}.release();

  auto const scatter_map = int32s_col{-2, -1, 0, 1, 2, 3}.release();
  test_scatter(structs_src, structs_tgt, structs_expected, scatter_map);
}

TYPED_TEST(TypedStructScatterTest, SimpleScatterTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto child_col_src =
    col_wrapper{{0, 1, 2, 3, null, XXX},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto const structs_src = structs_col{
    {child_col_src}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();

  // Target data
  auto child_col_tgt =
    col_wrapper{{50, null, 70, XXX, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_tgt = structs_col{
    {child_col_tgt}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
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
  test_scatter(structs_src, structs_tgt, structs_expected1, scatter_map1);

  // Expected data
  auto child_col_expected2 =
    col_wrapper{{1, null, 70, 3, 0, 2},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto const structs_expected2 = structs_col{
    {child_col_expected2}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return true;
    })}.release();
  auto const scatter_map2 = int32s_col{-2, 0, 5, 3}.release();
  test_scatter(structs_src, structs_tgt, structs_expected2, scatter_map2);
}

TYPED_TEST(TypedStructScatterTest, ComplexDataScatterTest)
{
  // Testing scatter() on struct<string, numeric, bool>.
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto names_column_src =
    strings_col{{"Newton", "Washington", "Cherry", "Kiwi", "Lemon", "Tomato"},
                cudf::detail::make_counting_transform_iterator(0, [](auto) { return true; })};
  auto ages_column_src =
    col_wrapper{{5, 10, 15, 20, 25, 30},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};
  auto is_human_col_src =
    bools_col{{true, true, false, false, false, false},
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; })};

  // Target data
  auto names_column_tgt =
    strings_col{{"String 0", "String 1", "String 2", "String 3", "String 4", "String 5"},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; })};
  auto ages_column_tgt =
    col_wrapper{{50, 60, 70, 80, 90, 100},
                cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })};
  auto is_human_col_tgt =
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

  auto const structs_src = structs_col{
    {names_column_src, ages_column_src, is_human_col_src},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 5;
    })}.release();
  auto const structs_tgt = structs_col{
    {names_column_tgt, ages_column_tgt, is_human_col_tgt},
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
  test_scatter(structs_src, structs_tgt, structs_expected, scatter_map);
}

TYPED_TEST(TypedStructScatterTest, ScatterStructOfListsTest)
{
  // Testing gather() on struct<list<numeric>>
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto lists_col_src =
    lists_col{{{5}, {10, 15}, {20, 25, 30}, {35, 40, 45, 50}, {55, 60, 65}, {70, 75}, {80}, {}, {}},
              // Valid for elements 0, 3, 6,...
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return !(i % 3); })};
  auto const structs_src = structs_col{{lists_col_src}}.release();

  // Target data
  auto lists_col_tgt =
    lists_col{{{1}, {2, 3}, {4, 5, 6}, {7, 8}, {9}, {10, 11, 12, 13}, {}, {14}, {15, 16}},
              // Valid for elements 1, 3, 5, 7,...
              cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })};
  auto const structs_tgt = structs_col{{lists_col_tgt}}.release();

  // Expected data
  auto const validity_expected = std::vector<bool>{0, 1, 1, 0, 0, 1, 1, 0, 0};
  auto lists_col_expected      = lists_col{
    {{1}, {2, 3}, {80}, {70, 75}, {55, 60, 65}, {35, 40, 45, 50}, {5}, {10, 15}, {20, 25, 30}},
    validity_expected.begin()};
  auto const structs_expected = structs_col{{lists_col_expected}}.release();

  // The first 2 elements of the target is not overwritten
  auto const scatter_map = int32s_col{-3, -2, -1, 5, 4, 3, 2}.release();
  test_scatter(structs_src, structs_tgt, structs_expected, scatter_map);
}
