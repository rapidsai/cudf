/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>

#include <functional>
#include <initializer_list>
#include <iterator>
#include <algorithm>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/column/column_factories.hpp"
#include "cudf/detail/utilities/device_operators.cuh"
#include "cudf/table/table_view.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/error.hpp"
#include "gtest/gtest.h"
#include "rmm/device_buffer.hpp"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/scan.h"
#include "thrust/sequence.h"

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using cudf::size_type;

struct StructColumnWrapperTest : public cudf::test::BaseFixture
{};

template<typename T>
struct TypedStructColumnWrapperTest : public cudf::test::BaseFixture
{};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedStructColumnWrapperTest, FixedWidthTypesNotBool);

// Test simple struct construction without nullmask, through column factory.
// Columns must retain their originally set values.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnFactoryConstruction)
{

  auto names_col = cudf::test::strings_column_wrapper{
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald"
  }.release();

  int num_rows {names_col->size()};

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<TypeParam>{
      {48, 27, 25} 
    }.release();
    
  auto is_human_col =
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false}
    }.release();

  vector_of_columns cols;
  cols.push_back(std::move(names_col));
  cols.push_back(std::move(ages_col));
  cols.push_back(std::move(is_human_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view {struct_col->view()};
  EXPECT_TRUE(
    std::all_of(
      struct_col_view.child_begin(), 
      struct_col_view.child_end(), 
      [&](auto const& child) {
        return child.size() == num_rows;
      }
    )
  );

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(
    cudf::test::strings_column_wrapper{
      "Samuel Vimes",
      "Carrot Ironfoundersson",
      "Angua von Uberwald"
    }.release()
  );
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<TypeParam>{
    48, 27, 25
  }.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<bool>{
    true, true, false
  }.release());

  std::for_each(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0)+expected_children.size(),
    [&](auto idx) {
      cudf::test::expect_columns_equal(
        struct_col_view.child(idx), 
        expected_children[idx]->view()
      );
    }
  );
}


// Test simple struct construction with nullmasks, through column wrappers.
// When the struct row is null, the child column value must be null.
TYPED_TEST(TypedStructColumnWrapperTest, TestColumnWrapperConstruction)
{
  std::initializer_list<std::string> names = {
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald",
    "Cheery Littlebottom",
    "Detritus", 
    "Mr Slant"
  };

  auto num_rows {std::distance(names.begin(), names.end())};

  auto names_col = cudf::test::strings_column_wrapper{
    names.begin(),
    names.end()
  };

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<TypeParam>{
      {48, 27, 25, 31, 351, 351}, 
      { 1,  1,  1,  1,   1,   0}
    };
    
  auto is_human_col =
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false},
      {   1,    1,     0,     1,     1,     0}
    };

  auto struct_col = 
    cudf::test::structs_column_wrapper{ 
      {names_col, ages_col, is_human_col}, 
      {1, 1, 1, 0, 1, 1}
    }.release();

  EXPECT_EQ(num_rows, struct_col->size());

  auto struct_col_view {struct_col->view()};
  EXPECT_TRUE(
    std::all_of(
      struct_col_view.child_begin(), 
      struct_col_view.child_end(), 
      [&](auto const& child) {
        return child.size() == num_rows;
      }
    )
  );

  // Check child columns for exactly correct values.
  vector_of_columns expected_children;
  expected_children.emplace_back(
    cudf::test::strings_column_wrapper{
      names,
      {1, 1, 1, 0, 1, 1}
    }.release()
  );
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<TypeParam>{
    {48, 27, 25, 31, 351, 351},
    { 1,  1,  1,  0,   1,   0} 
  }.release());
  expected_children.emplace_back(cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false},
    {   1,    1,     0,     0,     1,     0}
  }.release());

  std::for_each(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0)+expected_children.size(),
    [&](auto idx) {
      cudf::test::expect_columns_equal(
        struct_col_view.child(idx), 
        expected_children[idx]->view()
      );
    }
  );

  auto expected_struct_col =
    cudf::test::structs_column_wrapper{std::move(expected_children), {1, 1, 1, 0, 1, 1}}.release();

  cudf::test::expect_columns_equal(struct_col_view, expected_struct_col->view()); 
}


TYPED_TEST(TypedStructColumnWrapperTest, TestStructsContainingLists)
{
  // Test structs with two members:
  //  1. Name: String
  //  2. List: List<TypeParam>

  std::initializer_list<std::string> names = {
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald",
    "Cheery Littlebottom",
    "Detritus", 
    "Mr Slant"
  };

  auto num_rows {std::distance(names.begin(), names.end())}; 

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  // `List` member.
  auto lists_col = cudf::test::lists_column_wrapper<TypeParam>{
      {1,2,3},
      {4},
      {5,6},
      {},
      {7,8},
      {9}
  };

  // Construct a Struct column of 6 rows, with the last two values set to null.
  auto struct_col = cudf::test::structs_column_wrapper{
    {names_col, lists_col}, 
    {1, 1, 1, 1, 0, 0}
  }.release();

  // Check that the last two rows are null for all members.
  
  // For `Name` member, indices 4 and 5 are null.
  auto expected_names_col = cudf::test::strings_column_wrapper{
    names.begin(), 
    names.end(),
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return i<4; } )
  }.release();

  cudf::test::expect_columns_equal(struct_col->view().child(0), expected_names_col->view());
  
  // For the `List` member, indices 4, 5 should be null.
  // FIXME:  The way list columns are currently compared is not ideal for testing
  //         structs' list members. Rather than comparing for equivalence, 
  //         column_comparator_impl<list_view> currently checks that list's data (child)
  //         and offsets match perfectly.
  //         This causes two "equivalent lists" to compare unequal, if the data columns
  //         have different values at an index where the value is null.
  auto expected_last_two_lists_col = cudf::test::lists_column_wrapper<TypeParam>{
    {
      {1,2,3},
      {4},
      {5,6},
      {},
      {7,8}, // Null.
      {9}    // Null.
    },
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return i==0; })
  }.release();
  
  // FIXME: Uncomment after list comparison is fixed.
  // cudf::test::expect_columns_equal(
  //  struct_col->view().child(1), 
  //  expected_last_two_lists_col->view());
}


TYPED_TEST(TypedStructColumnWrapperTest, StructOfStructs)
{
  // Struct<is_human:bool, Struct<names:string, ages:int>>

  auto names = {
    "Samuel Vimes",
    "Carrot Ironfoundersson",
    "Angua von Uberwald",
    "Cheery Littlebottom",
    "Detritus", 
    "Mr Slant"
  };

  auto num_rows {std::distance(names.begin(), names.end())};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col = 
    cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, 
      { 1,  1,  1,  1,   1,   0}
    };

  auto struct_1 = cudf::test::structs_column_wrapper{
    {names_col, ages_col},
    {1, 1, 1, 1, 0, 1}
  };

  auto is_human_col =
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false},
      {   1,    1,     0,     1,     1,     0}
    }; 

  auto struct_2 = cudf::test::structs_column_wrapper{
    {is_human_col, struct_1},
    {0, 1, 1, 1, 1, 1}
  }.release();

  // Verify that the child/grandchild columns are as expected.
  auto expected_names_col = cudf::test::strings_column_wrapper(
    names.begin(), 
    names.end(), 
    cudf::test::make_counting_transform_iterator(0, [](auto i){ return i!=0 && i!=4; })).release();

  cudf::test::expect_columns_equal(*expected_names_col, struct_2->child(1).child(0));

  auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, 
    { 0,  1,  1,  1,   0,   0}
  }.release();
  cudf::test::expect_columns_equal(*expected_ages_col, struct_2->child(1).child(1));

  auto expected_bool_col = cudf::test::fixed_width_column_wrapper<bool> {
    {true, true, false, false, false, false},
    {   0,    1,     0,     1,     1,     0}
  }.release();

  cudf::test::expect_columns_equal(*expected_bool_col, struct_2->child(0));

  // Verify that recursive struct columns may be compared 
  // using expect_columns_equal.

  vector_of_columns expected_cols_1;
  expected_cols_1.emplace_back(std::move(expected_names_col));
  expected_cols_1.emplace_back(std::move(expected_ages_col));
  auto expected_struct_1 = cudf::test::structs_column_wrapper(std::move(expected_cols_1), {1, 1, 1, 1, 0, 1}).release();

  vector_of_columns expected_cols_2;
  expected_cols_2.emplace_back(std::move(expected_bool_col));
  expected_cols_2.emplace_back(std::move(expected_struct_1));
  auto expected_struct_2 = cudf::test::structs_column_wrapper(std::move(expected_cols_2), {0, 1, 1, 1, 1, 1}).release();

  cudf::test::expect_columns_equal(*expected_struct_2, *struct_2);
}


TEST_F(StructColumnWrapperTest, SimpleTestExpectStructColumnsEqual)
{
  auto ints_col = cudf::test::fixed_width_column_wrapper<int32_t>{{0,1}, {0,0}}.release();

  vector_of_columns cols;
  cols.emplace_back(std::move(ints_col));
  auto structs_col = cudf::test::structs_column_wrapper{std::move(cols)};
  
  cudf::test::expect_columns_equal(structs_col, structs_col);
}


CUDF_TEST_PROGRAM_MAIN()
