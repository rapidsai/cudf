/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/exec_policy.hpp>

using namespace cudf;

template <typename T>
struct RowBitCountTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(RowBitCountTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(RowBitCountTyped, SimpleTypes)
{
  using T = TypeParam;

  auto col = cudf::make_fixed_width_column(data_type{type_to_id<T>()}, 16);

  table_view t({*col});
  auto result = cudf::row_bit_count(t);

  // expect size of the type per row
  auto expected = make_fixed_width_column(data_type{type_id::INT32}, 16);
  cudf::mutable_column_view mcv(*expected);
  thrust::fill(rmm::exec_policy(0),
               mcv.begin<size_type>(),
               mcv.end<size_type>(),
               sizeof(device_storage_type_t<T>) * CHAR_BIT);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result);
}

TYPED_TEST(RowBitCountTyped, SimpleTypesWithNulls)
{
  using T = TypeParam;

  auto iter   = thrust::make_counting_iterator(0);
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](int i) { return i % 2 == 0 ? true : false; });
  cudf::test::fixed_width_column_wrapper<T> col(iter, iter + 16, valids);

  table_view t({col});
  auto result = cudf::row_bit_count(t);

  // expect size of the type + 1 bit per row
  auto expected = make_fixed_width_column(data_type{type_id::INT32}, 16);
  cudf::mutable_column_view mcv(*expected);
  thrust::fill(rmm::exec_policy(0),
               mcv.begin<size_type>(),
               mcv.end<size_type>(),
               (sizeof(device_storage_type_t<T>) * CHAR_BIT) + 1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result);
}

template <typename T>
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> build_list_column()
{
  using LCW                     = cudf::test::lists_column_wrapper<T, int>;
  constexpr size_type type_size = sizeof(device_storage_type_t<T>) * CHAR_BIT;

  // clang-format off
  cudf::test::lists_column_wrapper<T, int> col{ {{1, 2}, {3, 4, 5}}, 
                                                LCW{LCW{}}, 
                                                {LCW{10}},
                                                {{6, 7, 8}, {9}},
                                                {{-1, -2}, {-3, -4}},
                                                {{-5, -6, -7}, {-8, -9}} };
  // clang-format on

  // expected size = (num rows at level 1 + num_rows at level 2) + # values in the leaf
  cudf::test::fixed_width_column_wrapper<size_type> expected{
    ((4 + 8) * CHAR_BIT) + (type_size * 5),
    ((4 + 0) * CHAR_BIT) + (type_size * 0),
    ((4 + 4) * CHAR_BIT) + (type_size * 1),
    ((4 + 8) * CHAR_BIT) + (type_size * 4),
    ((4 + 8) * CHAR_BIT) + (type_size * 4),
    ((4 + 8) * CHAR_BIT) + (type_size * 5)};

  return {col.release(), expected.release()};
}

TYPED_TEST(RowBitCountTyped, Lists)
{
  using T = TypeParam;

  std::unique_ptr<column> col;
  std::unique_ptr<column> expected_sizes;
  std::tie(col, expected_sizes) = build_list_column<T>();

  table_view t({*col});
  auto result = cudf::row_bit_count(t);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_sizes, *result);
}

TYPED_TEST(RowBitCountTyped, ListsWithNulls)
{
  using T                       = TypeParam;
  using LCW                     = cudf::test::lists_column_wrapper<T, int>;
  constexpr size_type type_size = sizeof(device_storage_type_t<T>) * CHAR_BIT;

  std::vector<bool> valids{true, false, true};
  std::vector<bool> valids2{false, true, false};
  std::vector<bool> valids3{true, false};

  // clang-format off
  cudf::test::lists_column_wrapper<T, int> col{ {{1, 2}, {{3, 4, 5}, valids.begin()}}, 
                                                LCW{LCW{}}, 
                                                {LCW{10}}, 
                                                {{{{6, 7, 8}, valids2.begin()}, {9}}, valids3.begin()} };
  // clang-format on

  table_view t({col});
  auto result = cudf::row_bit_count(t);

  // expected size = (num rows at level 1 + num_rows at level 2) + # values in the leaf + validity
  // where applicable
  cudf::test::fixed_width_column_wrapper<size_type> expected{
    ((4 + 8) * CHAR_BIT) + (type_size * 5) + 7,
    ((4 + 0) * CHAR_BIT) + (type_size * 0),
    ((4 + 4) * CHAR_BIT) + (type_size * 1) + 2,
    ((4 + 8) * CHAR_BIT) + (type_size * 3) + 5};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

struct RowBitCount : public cudf::test::BaseFixture {
};

TEST_F(RowBitCount, Strings)
{
  std::vector<std::string> strings{"abc", "ï", "", "z", "bananas", "warp", "", "zing"};

  cudf::test::strings_column_wrapper col(strings.begin(), strings.end());

  table_view t({col});
  auto result = cudf::row_bit_count(t);

  // expect 1 offset (4 bytes) + length of string per row
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, [&strings](int i) {
    return (static_cast<size_type>(strings[i].size()) + sizeof(offset_type)) * CHAR_BIT;
  });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter, size_iter + strings.size());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, StringsWithNulls)
{
  // clang-format off
  std::vector<std::string> strings { "daïs", "def", "", "z", "bananas", "warp", "", "zing" };
  std::vector<bool>        valids  {  1,      0,    0,  1,   0,          1,      1,  1 };
  // clang-format on

  cudf::test::strings_column_wrapper col(strings.begin(), strings.end(), valids.begin());

  table_view t({col});
  auto result = cudf::row_bit_count(t);

  // expect 1 offset (4 bytes) + (length of string, or 0 if null) + 1 validity bit per row
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, [&strings, &valids](int i) {
    return ((static_cast<size_type>(valids[i] ? strings[i].size() : 0) + sizeof(offset_type)) *
            CHAR_BIT) +
           1;
  });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter, size_iter + strings.size());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> build_struct_column()
{
  std::vector<bool> struct_validity{0, 1, 1, 1, 1, 0};
  std::vector<std::string> strings{"abc", "def", "", "z", "bananas", "daïs"};

  cudf::test::fixed_width_column_wrapper<float> col0{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{8, 9, 10, 11, 12, 13}, {1, 0, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col2(strings.begin(), strings.end());

  // creating a struct column will cause all child columns to be promoted to have validity
  cudf::test::structs_column_wrapper struct_col({col0, col1, col2}, struct_validity);

  // expect (1 offset (4 bytes) + (length of string if row is valid) + 1 validity bit) +
  //        (1 float + 1 validity bit) +
  //        (1 int16_t + 1 validity bit) +
  //        (1 validity bit)
  auto size_iter =
    cudf::detail::make_counting_transform_iterator(0, [&strings, &struct_validity](int i) {
      return (sizeof(float) * CHAR_BIT) + 1 + (sizeof(int16_t) * CHAR_BIT) + 1 +
             (static_cast<size_type>(strings[i].size()) * CHAR_BIT) +
             (sizeof(offset_type) * CHAR_BIT) + 1 + 1;
    });
  cudf::test::fixed_width_column_wrapper<size_type> expected_sizes(size_iter,
                                                                   size_iter + strings.size());

  return {struct_col.release(), expected_sizes.release()};
}

TEST_F(RowBitCount, StructsNoNulls)
{
  std::vector<std::string> strings{"abc", "daïs", "", "z", "bananas", "warp"};

  cudf::test::fixed_width_column_wrapper<float> col0{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<int16_t> col1{8, 9, 10, 11, 12, 13};
  cudf::test::strings_column_wrapper col2(strings.begin(), strings.end());

  cudf::test::structs_column_wrapper struct_col({col0, col1, col2});

  table_view t({struct_col});
  auto result = cudf::row_bit_count(t);

  // expect 1 offset (4 bytes) + (length of string) + 1 float + 1 int16_t
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, [&strings](int i) {
    return ((sizeof(float) + sizeof(int16_t)) * CHAR_BIT) +
           ((static_cast<size_type>(strings[i].size()) + sizeof(offset_type)) * CHAR_BIT);
  });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter, size_iter + t.num_rows());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, StructsNulls)
{
  std::unique_ptr<column> struct_col;
  std::unique_ptr<column> expected_sizes;
  std::tie(struct_col, expected_sizes) = build_struct_column();
  table_view t({*struct_col});
  auto result = cudf::row_bit_count(t);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_sizes, *result);
}

TEST_F(RowBitCount, StructsNested)
{
  // struct<struct<int>, int16>
  cudf::test::fixed_width_column_wrapper<int> col0{0, 1, 2, 3, 4, 5};
  cudf::test::structs_column_wrapper inner_struct({col0});

  cudf::test::fixed_width_column_wrapper<int16_t> col1{8, 9, 10, 11, 12, 13};
  cudf::test::structs_column_wrapper struct_col({inner_struct, col1});

  table_view t({struct_col});
  auto result = cudf::row_bit_count(t);

  // expect num_rows * (4 + 2) bytes
  auto size_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int i) { return (4 + 2) * CHAR_BIT; });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter, size_iter + t.num_rows());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> build_nested_and_expected_column(
  std::vector<bool> const& struct_validity)
{
  // tests the "branching" case ->  list<struct<list> ...>>>

  // List<Struct<List<int>, float, int16>

  // Inner list column
  // clang-format off
  cudf::test::lists_column_wrapper<int> list{
    {1, 2, 3, 4, 5},     
    {6, 7, 8},
    {33, 34, 35, 36, 37, 38, 39},
    {-1, -2},
    {-10, -11, -1, -20},
    {40, 41, 42},
    {100, 200, 300},
    {-100, -200, -300}};
  // clang-format on

  // floats
  std::vector<float> ages{5, 10, 15, 20, 4, 75, 16, -16};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 1};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<float>(ages.begin(), ages.end(), ages_validity.begin());

  // int16 values
  std::vector<int16_t> vals{-1, -2, -3, 1, 2, 3, 8, 9};
  auto i16_column = cudf::test::fixed_width_column_wrapper<int16_t>(vals.begin(), vals.end());

  // Assemble struct column
  auto struct_column =
    cudf::test::structs_column_wrapper({list, ages_column, i16_column}, struct_validity);

  // wrap in a list
  std::vector<int> outer_offsets{0, 1, 1, 3, 6, 7, 8};
  cudf::test::fixed_width_column_wrapper<int> outer_offsets_col(outer_offsets.begin(),
                                                                outer_offsets.end());
  auto const size = static_cast<column_view>(outer_offsets_col).size() - 1;

  cudf::test::fixed_width_column_wrapper<size_type> expected_sizes{276, 32, 520, 572, 212, 212};

  return {cudf::make_lists_column(static_cast<cudf::size_type>(size),
                                  outer_offsets_col.release(),
                                  struct_column.release(),
                                  cudf::UNKNOWN_NULL_COUNT,
                                  rmm::device_buffer{}),
          expected_sizes.release()};
}

std::unique_ptr<column> build_nested_column(std::vector<bool> const& struct_validity)
{
  // List<Struct<List<List<int>>, Struct<int16>>>

  // Inner list column
  // clang-format off
  cudf::test::lists_column_wrapper<int> list{    
     {{1, 2, 3, 4, 5}, {2, 3}},
     {{6, 7, 8}, {8, 9}},
     {{1, 2}, {3, 4, 5}, {33, 34, 35, 36, 37, 38, 39}}};
  // clang-format on

  // Inner struct
  std::vector<int16_t> vals{-1, -2, -3};
  auto i16_column   = cudf::test::fixed_width_column_wrapper<int16_t>(vals.begin(), vals.end());
  auto inner_struct = cudf::test::structs_column_wrapper({i16_column});

  // outer struct
  auto outer_struct = cudf::test::structs_column_wrapper({list, inner_struct}, struct_validity);

  // wrap in a list
  std::vector<int> outer_offsets{0, 1, 1, 3};
  cudf::test::fixed_width_column_wrapper<int> outer_offsets_col(outer_offsets.begin(),
                                                                outer_offsets.end());
  auto const size = static_cast<column_view>(outer_offsets_col).size() - 1;
  return make_lists_column(static_cast<cudf::size_type>(size),
                           outer_offsets_col.release(),
                           outer_struct.release(),
                           cudf::UNKNOWN_NULL_COUNT,
                           rmm::device_buffer{});
}

TEST_F(RowBitCount, NestedTypes)
{
  // List<Struct<List<int>, float, List<int>, int16>
  {
    std::unique_ptr<column> col_no_nulls;
    std::unique_ptr<column> expected_sizes;
    std::tie(col_no_nulls, expected_sizes) =
      build_nested_and_expected_column({1, 1, 1, 1, 1, 1, 1, 1});
    table_view no_nulls_t({*col_no_nulls});
    auto no_nulls_result = cudf::row_bit_count(no_nulls_t);

    auto col_nulls = build_nested_and_expected_column({0, 0, 1, 1, 1, 1, 1, 1}).first;
    table_view nulls_t({*col_nulls});
    auto nulls_result = cudf::row_bit_count(nulls_t);

    // List<Struct<List<int>, float, int16>
    //
    // this illustrates the difference between a row_bit_count
    // returning a pre-gather result, or a post-gather result.
    //
    // in a post-gather situation, the nulls in the struct would result in the values
    // nested in the list below to be dropped, resulting in smaller row sizes.
    //
    // however, for performance reasons, row_bit_count simply walks the data that is
    // currently there. so list rows that are null, but have a real span of
    // offsets (X, Y) instead of (X, X)  will end up getting the child data for those
    // rows included.
    //
    // if row_bit_count() is changed to return a post-gather result (which may be desirable),
    // the nulls_result case below will start failing and will need to be changed.
    //
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_sizes, *no_nulls_result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_sizes, *nulls_result);
  }

  // List<Struct<List<List<int>>, Struct<int16>>>
  {
    auto col_no_nulls = build_nested_column({1, 1, 1});
    table_view no_nulls_t({*col_no_nulls});
    auto no_nulls_result = cudf::row_bit_count(no_nulls_t);

    auto col_nulls = build_nested_column({1, 0, 1});
    table_view nulls_t({*col_nulls});
    auto nulls_result = cudf::row_bit_count(nulls_t);

    cudf::test::fixed_width_column_wrapper<size_type> expected_sizes{372, 32, 840};

    // same explanation as above
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sizes, *no_nulls_result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sizes, *nulls_result);
  }

  // test pushing/popping multiple times within one struct, and branch depth > 1
  //
  // Struct<int, List<int>, float, List<List<int16>>, Struct<List<int>, List<Struct<List<int>,
  // float>>, int8_t>>
  {
    cudf::test::lists_column_wrapper<int> l0{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {5}};
    cudf::test::lists_column_wrapper<int16_t> l1{
      {{-1, -2}, {3, 4}}, {{4, 5}, {6, 7, 8}}, {{-6, -7}, {2}}, {{-11, -11}, {-12, -12}, {3}}};
    cudf::test::lists_column_wrapper<int> l2{{-1, -2}, {4, 5}, {-6, -7}, {1}};
    cudf::test::lists_column_wrapper<int> l3{{-1, -2, 0}, {5}, {-1, -6, -7}, {1, 2}};

    cudf::test::fixed_width_column_wrapper<int> c0{1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> c1{1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<int8_t> c2{1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> c3{11, 12, 13, 14};

    // innermost List<Struct<List<int>>>
    auto innermost_struct = cudf::test::structs_column_wrapper({l3, c3});
    std::vector<int> l4_offsets{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<int> l4_offsets_col(l4_offsets.begin(),
                                                               l4_offsets.end());
    auto const l4_size = l4_offsets.size() - 1;
    auto l4            = cudf::make_lists_column(static_cast<cudf::size_type>(l4_size),
                                      l4_offsets_col.release(),
                                      innermost_struct.release(),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      rmm::device_buffer{});

    // inner struct
    std::vector<std::unique_ptr<column>> inner_struct_children;
    inner_struct_children.push_back(l2.release());
    inner_struct_children.push_back(std::move(l4));
    auto inner_struct = cudf::test::structs_column_wrapper(std::move(inner_struct_children));

    // outer struct
    auto struct_col = cudf::test::structs_column_wrapper({c0, l0, c1, l1, inner_struct, c2});

    table_view t({struct_col});
    auto result = cudf::row_bit_count(t);

    cudf::test::fixed_width_column_wrapper<size_type> expected_sizes{648, 568, 664, 568};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sizes, *result);
  }
}

struct sum_functor {
  size_type const* s0;
  size_type const* s1;
  size_type const* s2;

  size_type operator() __device__(int i) { return s0[i] + s1[i] + s2[i]; }
};

TEST_F(RowBitCount, Table)
{
  // complex nested column
  std::unique_ptr<column> col0;
  std::unique_ptr<column> col0_sizes;
  std::tie(col0, col0_sizes) = build_nested_and_expected_column({1, 1, 1, 1, 1, 1, 1, 1});

  // struct column
  std::unique_ptr<column> col1;
  std::unique_ptr<column> col1_sizes;
  std::tie(col1, col1_sizes) = build_struct_column();

  // list column
  std::unique_ptr<column> col2;
  std::unique_ptr<column> col2_sizes;
  std::tie(col2, col2_sizes) = build_list_column<int16_t>();

  table_view t({*col0, *col1, *col2});
  auto result = cudf::row_bit_count(t);

  // sum all column sizes
  column_view cv0 = static_cast<column_view>(*col0_sizes);
  column_view cv1 = static_cast<column_view>(*col1_sizes);
  column_view cv2 = static_cast<column_view>(*col2_sizes);
  auto expected   = cudf::make_fixed_width_column(data_type{type_id::INT32}, t.num_rows());
  cudf::mutable_column_view mcv(*expected);
  thrust::transform(
    rmm::exec_policy(0),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + t.num_rows(),
    mcv.begin<size_type>(),
    sum_functor{cv0.data<size_type>(), cv1.data<size_type>(), cv2.data<size_type>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result);
}

TEST_F(RowBitCount, SlicedColumnsFixedWidth)
{
  auto const slice_size = 7;
  cudf::test::fixed_width_column_wrapper<int16_t> c0_unsliced{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto c0 = cudf::slice(c0_unsliced, {2, 2 + slice_size});

  table_view t({c0});
  auto result = cudf::row_bit_count(t);

  cudf::test::fixed_width_column_wrapper<size_type> expected{16, 16, 16, 16, 16, 16, 16};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, SlicedColumnsStrings)
{
  auto const slice_size = 7;
  std::vector<std::string> strings{
    "banana", "metric", "imperial", "abc", "daïs", "", "fire", "def", "cudf", "xyzw"};
  cudf::test::strings_column_wrapper c0_unsliced(strings.begin(), strings.end());
  auto c0 = cudf::slice(c0_unsliced, {3, 3 + slice_size});

  table_view t({c0});
  auto result = cudf::row_bit_count(t);

  // expect 1 offset (4 bytes) + length of string per row
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, [&strings](int i) {
    return (static_cast<size_type>(strings[i].size()) + sizeof(offset_type)) * CHAR_BIT;
  });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter + 3,
                                                             size_iter + 3 + slice_size);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, SlicedColumnsLists)
{
  auto const slice_size = 2;
  cudf::test::lists_column_wrapper<cudf::string_view> c0_unsliced{
    {{"banana", "v"}, {"cats"}},
    {{"dogs", "yay"}, {"xyz", ""}, {"daïs"}},
    {{"fast", "parrot"}, {"orange"}},
    {{"blue"}, {"red", "yellow"}, {"ultraviolet", "", "green"}}};
  auto c0 = cudf::slice(c0_unsliced, {1, 1 + slice_size});

  table_view t({c0});
  auto result = cudf::row_bit_count(t);

  cudf::test::fixed_width_column_wrapper<size_type> expected{408, 320};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, SlicedColumnsStructs)
{
  auto const slice_size = 7;

  cudf::test::fixed_width_column_wrapper<int16_t> c0{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<std::string> strings{
    "banana", "metric", "imperial", "abc", "daïs", "", "fire", "def", "cudf", "xyzw"};
  cudf::test::strings_column_wrapper c1(strings.begin(), strings.end());

  auto struct_col_unsliced = cudf::test::structs_column_wrapper({c0, c1});
  auto struct_col          = cudf::slice(struct_col_unsliced, {3, 3 + slice_size});

  table_view t({struct_col});
  auto result = cudf::row_bit_count(t);

  // expect 1 offset (4 bytes) + length of string per row + 1 int16_t per row
  auto size_iter = cudf::detail::make_counting_transform_iterator(0, [&strings](int i) {
    return (static_cast<size_type>(strings[i].size()) + sizeof(offset_type) + sizeof(int16_t)) *
           CHAR_BIT;
  });
  cudf::test::fixed_width_column_wrapper<size_type> expected(size_iter + 3,
                                                             size_iter + 3 + slice_size);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RowBitCount, EmptyTable)
{
  {
    cudf::table_view empty;
    auto result = cudf::row_bit_count(empty);
    CUDF_EXPECTS(result != nullptr && result->size() == 0, "Expected an empty column");
  }

  {
    auto strings = cudf::strings::detail::make_empty_strings_column(0);
    auto ints    = cudf::make_empty_column(data_type{type_id::INT32});
    cudf::table_view empty({*strings, *ints});

    auto result = cudf::row_bit_count(empty);
    CUDF_EXPECTS(result != nullptr && result->size() == 0, "Expected an empty column");
  }
}