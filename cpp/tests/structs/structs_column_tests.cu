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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using cudf::size_type;

struct StructColumnWrapperTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedStructColumnWrapperTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedStructColumnWrapperTest, FixedWidthTypesNotBool);

// Test simple struct construction without nullmask, through column factory.
// Columns must retain their originally set values.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnFactoryConstruction)
{
  auto names_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Uberwald"}
      .release();

  int num_rows{names_col->size()};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{{48, 27, 25}}.release();

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}}.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));
  cols.push_back(std::move(is_human_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view{struct_col->view()};
  EXPECT_TRUE(std::all_of(struct_col_view.child_begin(),
                          struct_col_view.child_end(),
                          [&](auto const& child) { return child.size() == num_rows; }));

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(cudf::test::strings_column_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Uberwald"}
                                   .release());
  expected_children.emplace_back(
    cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{48, 27, 25}.release());
  expected_children.emplace_back(
    cudf::test::fixed_width_column_wrapper<bool>{true, true, false}.release());

  std::for_each(thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + expected_children.size(),
                [&](auto idx) {
                  cudf::test::expect_columns_equivalent(struct_col_view.child(idx),
                                                        expected_children[idx]->view());
                });
}

// Test simple struct construction with nullmasks, through column wrappers.
// When the struct row is null, the child column value must be null.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnWrapperConstruction)
{
  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Uberwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};

  auto num_rows{std::distance(names.begin(), names.end())};

  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_col =
    cudf::test::structs_column_wrapper{{names_col, ages_col, is_human_col}, {1, 1, 1, 0, 1, 1}}
      .release();

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view{struct_col->view()};
  EXPECT_TRUE(std::all_of(struct_col_view.child_begin(),
                          struct_col_view.child_end(),
                          [&](auto const& child) { return child.size() == num_rows; }));

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(
    cudf::test::strings_column_wrapper{names, {1, 1, 1, 0, 1, 1}}.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {48, 27, 25, 31, 351, 351},
    {1, 1, 1, 0, 1, 0}}.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false},
    {1, 1, 0, 0, 1, 0}}.release());

  std::for_each(thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + expected_children.size(),
                [&](auto idx) {
                  cudf::test::expect_columns_equivalent(struct_col_view.child(idx),
                                                        expected_children[idx]->view());
                });

  auto expected_struct_col =
    cudf::test::structs_column_wrapper{std::move(expected_children), {1, 1, 1, 0, 1, 1}}.release();

  cudf::test::expect_columns_equivalent(struct_col_view, expected_struct_col->view());
}

TYPED_TEST(TypedStructColumnWrapperTest, TestStructsContainingLists)
{
  // Test structs with two members:
  //  1. Name: String
  //  2. List: List<TypeParam>

  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Uberwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};

  auto num_rows{std::distance(names.begin(), names.end())};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  // `List` member.
  auto lists_col =
    cudf::test::lists_column_wrapper<TypeParam, int32_t>{{1, 2, 3}, {4}, {5, 6}, {}, {7, 8}, {9}};

  // Construct a Struct column of 6 rows, with the last two values set to null.
  auto struct_col =
    cudf::test::structs_column_wrapper{{names_col, lists_col}, {1, 1, 1, 1, 0, 0}}.release();

  // Check that the last two rows are null for all members.

  // For `Name` member, indices 4 and 5 are null.
  auto expected_names_col = cudf::test::strings_column_wrapper{
    names.begin(), names.end(), cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i < 4;
    })}.release();

  cudf::test::expect_columns_equivalent(struct_col->view().child(0), expected_names_col->view());

  // For the `List` member, indices 4, 5 should be null.
  // FIXME:  The way list columns are currently compared is not ideal for testing
  //         structs' list members. Rather than comparing for equivalence,
  //         column_comparator_impl<list_view> currently checks that list's data (child)
  //         and offsets match perfectly.
  //         This causes two "equivalent lists" to compare unequal, if the data columns
  //         have different values at an index where the value is null.
  auto expected_last_two_lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>{
    {
      {1, 2, 3},
      {4},
      {5, 6},
      {},
      {7, 8},  // Null.
      {9}      // Null.
    },
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i == 0;
    })}.release();

  // FIXME: Uncomment after list comparison is fixed.
  // cudf::test::expect_columns_equivalent(
  //  struct_col->view().child(1),
  //  expected_last_two_lists_col->view());
}

TYPED_TEST(TypedStructColumnWrapperTest, StructOfStructs)
{
  // Struct<is_human:bool, Struct<names:string, ages:int>>

  auto names = {"Samuel Vimes",
                "Carrot Ironfoundersson",
                "Angua von Uberwald",
                "Cheery Littlebottom",
                "Detritus",
                "Mr Slant"};

  auto num_rows{std::distance(names.begin(), names.end())};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{names_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  // Verify that the child/grandchild columns are as expected.
  auto expected_names_col =
    cudf::test::strings_column_wrapper(
      names.begin(),
      names.end(),
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0 && i != 4; }))
      .release();

  cudf::test::expect_columns_equivalent(*expected_names_col, struct_2->child(1).child(0));

  auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351},
    {0, 1, 1, 1, 0, 0}}.release();
  cudf::test::expect_columns_equivalent(*expected_ages_col, struct_2->child(1).child(1));

  auto expected_bool_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false},
    {0, 1, 0, 1, 1, 0}}.release();

  cudf::test::expect_columns_equivalent(*expected_bool_col, struct_2->child(0));

  // Verify that recursive struct columns may be compared
  // using expect_columns_equivalent.

  vector_of_columns expected_cols_1;
  expected_cols_1.emplace_back(std::move(expected_names_col));
  expected_cols_1.emplace_back(std::move(expected_ages_col));
  auto expected_struct_1 =
    cudf::test::structs_column_wrapper(std::move(expected_cols_1), {1, 1, 1, 1, 0, 1}).release();

  vector_of_columns expected_cols_2;
  expected_cols_2.emplace_back(std::move(expected_bool_col));
  expected_cols_2.emplace_back(std::move(expected_struct_1));
  auto expected_struct_2 =
    cudf::test::structs_column_wrapper(std::move(expected_cols_2), {0, 1, 1, 1, 1, 1}).release();

  cudf::test::expect_columns_equivalent(*expected_struct_2, *struct_2);
}

TYPED_TEST(TypedStructColumnWrapperTest, TestNullMaskPropagationForNonNullStruct)
{
  // Struct<is_human:bool, Struct<names:string, ages:int>>

  auto names = {"Samuel Vimes",
                "Carrot Ironfoundersson",
                "Angua von Uberwald",
                "Cheery Littlebottom",
                "Detritus",
                "Mr Slant"};

  auto num_rows{std::distance(names.begin(), names.end())};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 1}  // <-- No nulls in ages_col either.
  };

  auto struct_1 = cudf::test::structs_column_wrapper{
    {names_col, ages_col}, {1, 1, 1, 1, 1, 1}  // <-- Non-null, bottom level struct.
  };

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{
      {is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}  // <-- First row is null, for top-level struct.
    }
      .release();

  // Verify that the child/grandchild columns are as expected.

  // Top-struct has 1 null (at index 0).
  // Bottom-level struct had no nulls, but must now report nulls
  auto expected_names_col =
    cudf::test::strings_column_wrapper(
      names.begin(),
      names.end(),
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; }))
      .release();

  cudf::test::expect_columns_equivalent(*expected_names_col, struct_2->child(1).child(0));

  auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351},
    {0, 1, 1, 1, 1, 1}}.release();
  cudf::test::expect_columns_equivalent(*expected_ages_col, struct_2->child(1).child(1));

  auto expected_bool_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false},
    {0, 1, 0, 1, 1, 0}}.release();

  cudf::test::expect_columns_equivalent(*expected_bool_col, struct_2->child(0));

  // Verify that recursive struct columns may be compared
  // using expect_columns_equivalent.

  vector_of_columns expected_cols_1;
  expected_cols_1.emplace_back(std::move(expected_names_col));
  expected_cols_1.emplace_back(std::move(expected_ages_col));
  auto expected_struct_1 =
    cudf::test::structs_column_wrapper(std::move(expected_cols_1), {1, 1, 1, 1, 1, 1}).release();

  vector_of_columns expected_cols_2;
  expected_cols_2.emplace_back(std::move(expected_bool_col));
  expected_cols_2.emplace_back(std::move(expected_struct_1));
  auto expected_struct_2 =
    cudf::test::structs_column_wrapper(std::move(expected_cols_2), {0, 1, 1, 1, 1, 1}).release();

  cudf::test::expect_columns_equivalent(*expected_struct_2, *struct_2);
}

TEST_F(StructColumnWrapperTest, StructWithNoMembers)
{
  auto struct_col{cudf::test::structs_column_wrapper{}.release()};
  EXPECT_TRUE(struct_col->num_children() == 0);
  EXPECT_TRUE(struct_col->null_count() == 0);
  EXPECT_TRUE(struct_col->size() == 0);
}

TYPED_TEST(TypedStructColumnWrapperTest, StructsWithMembersWithDifferentRowCounts)
{
  auto numeric_col_5 = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{{1, 2, 3, 4, 5}};
  auto bool_col_4    = cudf::test::fixed_width_column_wrapper<bool>{1, 0, 1, 0};

  EXPECT_THROW(cudf::test::structs_column_wrapper({numeric_col_5, bool_col_4}), cudf::logic_error);
}

TYPED_TEST(TypedStructColumnWrapperTest, TestListsOfStructs)
{
  // Test structs with two members:
  //  1. Name: String
  //  2. List: List<TypeParam>

  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Uberwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};

  auto num_rows{std::distance(names.begin(), names.end())};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  // Numeric column has some nulls.
  auto ages_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_col =
    cudf::test::structs_column_wrapper({names_col, ages_col}, {1, 1, 1, 0, 0, 1}).release();

  auto expected_unchanged_struct_col = cudf::column(*struct_col);

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto list_col = cudf::make_lists_column(num_list_rows,
                                          std::move(list_offsets_column),
                                          std::move(struct_col),
                                          cudf::UNKNOWN_NULL_COUNT,
                                          {});

  // List of structs was constructed successfully. No exceptions.
  // Verify that child columns is as it was set.

  cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                        cudf::lists_column_view(*list_col).child());

#ifndef NDEBUG
  std::cout << "Printing list col: \n";
  cudf::test::print(*list_col);
#endif
}

TYPED_TEST(TypedStructColumnWrapperTest, ListOfStructOfList)
{
  using namespace cudf::test;

  auto list_col = lists_column_wrapper<TypeParam, int32_t>{
    {{0}, {1}, {}, {3}, {4}, {5, 5}, {6}, {}, {8}, {9}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })};

  // TODO: Struct<List> cannot be compared with expect_columns_equal(),
  // if the struct has null values. After lists support "equivalence"
  // comparisons, the structs column needs to be modified to add nulls.
  auto struct_of_lists_col = structs_column_wrapper{{list_col}}.release();

  auto list_of_struct_of_list_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3; });
  auto list_of_struct_of_list = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<size_type>{0, 2, 4, 6, 8, 10}.release()),
    std::move(struct_of_lists_col),
    cudf::UNKNOWN_NULL_COUNT,
    detail::make_null_mask(list_of_struct_of_list_validity, list_of_struct_of_list_validity + 5));

  // Compare with expected values.

  auto expected_level0_list = lists_column_wrapper<TypeParam, int32_t>{
    {{}, {1}, {}, {3}, {}, {5, 5}, {}, {}, {}, {9}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })};

  auto expected_level2_struct = structs_column_wrapper{{expected_level0_list}}.release();

  expect_columns_equivalent(cudf::lists_column_view(*list_of_struct_of_list).child(),
                            *expected_level2_struct);

  auto expected_level3_list = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<size_type>{0, 2, 4, 6, 8, 10}.release()),
    std::move(expected_level2_struct),
    cudf::UNKNOWN_NULL_COUNT,
    detail::make_null_mask(list_of_struct_of_list_validity, list_of_struct_of_list_validity + 5));

  expect_columns_equivalent(*list_of_struct_of_list, *expected_level3_list);
}

TYPED_TEST(TypedStructColumnWrapperTest, StructOfListOfStruct)
{
  using namespace cudf::test;

  auto ints_col = fixed_width_column_wrapper<TypeParam, int32_t>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })};

  auto structs_col =
    structs_column_wrapper{
      {ints_col},
      cudf::detail::make_counting_transform_iterator(
        0, [](auto i) { return i < 6; })  // Last 4 structs are null.
    }
      .release();

  auto list_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3; });
  auto lists_col = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<size_type>{0, 2, 4, 6, 8, 10}.release()),
    std::move(structs_col),
    cudf::UNKNOWN_NULL_COUNT,
    detail::make_null_mask(list_validity, list_validity + 5));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(lists_col));
  auto struct_of_list_of_struct = structs_column_wrapper{std::move(cols)}.release();

  // Check that the struct is constructed as expected.

  auto expected_ints_col = fixed_width_column_wrapper<TypeParam, int32_t>{
    {0, 1, 0, 3, 0, 5, 0, 0, 0, 0}, {0, 1, 0, 1, 0, 1, 0, 0, 0, 0}};

  auto expected_structs_col =
    structs_column_wrapper{{expected_ints_col}, {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}}.release();

  auto expected_lists_col = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<size_type>{0, 2, 4, 6, 8, 10}.release()),
    std::move(expected_structs_col),
    cudf::UNKNOWN_NULL_COUNT,
    detail::make_null_mask(list_validity, list_validity + 5));

  // Test that the lists child column is as expected.
  cudf::test::expect_columns_equivalent(*expected_lists_col, struct_of_list_of_struct->child(0));

  // Test that the outer struct column is as expected.
  cols.clear();
  cols.push_back(std::move(expected_lists_col));
  cudf::test::expect_columns_equivalent(*(structs_column_wrapper{std::move(cols)}.release()),
                                        *struct_of_list_of_struct);
}

TYPED_TEST(TypedStructColumnWrapperTest, EmptyColumnsOfStructs)
{
  using namespace cudf::test;

  {
    // Empty struct column.
    auto empty_struct_column = structs_column_wrapper{}.release();
    EXPECT_TRUE(empty_struct_column->num_children() == 0);
    EXPECT_TRUE(empty_struct_column->size() == 0);
    EXPECT_TRUE(empty_struct_column->null_count() == 0);
  }

  {
    // Empty struct<list> column.
    auto empty_list_column = lists_column_wrapper<TypeParam>{};
    auto struct_column     = structs_column_wrapper{{empty_list_column}}.release();
    EXPECT_TRUE(struct_column->num_children() == 1);
    EXPECT_TRUE(struct_column->size() == 0);
    EXPECT_TRUE(struct_column->null_count() == 0);

    auto empty_list_of_structs =
      cudf::make_lists_column(0,
                              fixed_width_column_wrapper<size_type>{0}.release(),
                              std::move(struct_column),
                              cudf::UNKNOWN_NULL_COUNT,
                              {});

    EXPECT_TRUE(empty_list_of_structs->size() == 0);
    EXPECT_TRUE(empty_list_of_structs->null_count() == 0);

    auto child_struct_column = cudf::lists_column_view(*empty_list_of_structs).child();
    EXPECT_TRUE(child_struct_column.num_children() == 1);
    EXPECT_TRUE(child_struct_column.size() == 0);
    EXPECT_TRUE(child_struct_column.null_count() == 0);
  }

  // TODO: Uncomment test after adding support to compare empty
  //   lists whose child columns may not be empty.
  // {
  //   auto non_empty_column_of_numbers =
  //     fixed_width_column_wrapper<TypeParam>{1,2,3,4,5}.release();
  //
  //   auto list_offsets =
  //     fixed_width_column_wrapper<size_type>{0}.release();
  //
  //   auto empty_list_column =
  //     cudf::make_lists_column(
  //       0, std::move(list_offsets), std::move(non_empty_column_of_numbers), 0, {});
  //
  //   expect_columns_equivalent(*lists_column_wrapper<TypeParam>{}.release(), *empty_list_column);
  //   auto struct_column = structs_column_wrapper{{empty_list_column}}.release();
  //   EXPECT_TRUE(struct_column->num_children() == 1);
  //   EXPECT_TRUE(struct_column->size() == 0);
  //   EXPECT_TRUE(struct_column->null_count() == 0);
  // }
}

TYPED_TEST(TypedStructColumnWrapperTest, CopyColumnFromView)
{
  // Testing deep-copying structs from column-views.

  using namespace cudf::test;
  using T = TypeParam;

  auto numeric_column =
    fixed_width_column_wrapper<T, int32_t>{{0, 1, 2, 3, 4, 5}, {1, 1, 1, 1, 1, 0}};

  auto lists_column = lists_column_wrapper<T, int32_t>{
    {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })};

  auto structs_column = structs_column_wrapper{
    {numeric_column, lists_column},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; })};

  auto clone_structs_column = cudf::column(structs_column);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(clone_structs_column, structs_column);

  auto list_of_structs_column =
    cudf::make_lists_column(3,
                            fixed_width_column_wrapper<int32_t>{0, 2, 4, 6}.release(),
                            structs_column.release(),
                            cudf::UNKNOWN_NULL_COUNT,
                            {})
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_of_structs_column->view(),
                                      cudf::column(list_of_structs_column->view()));
}

CUDF_TEST_PROGRAM_MAIN()
