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
 *
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

struct ContainsTest : public cudf::test::BaseFixture {
};

using ContainsTestTypes = cudf::test::
  Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes, cudf::test::ChronoTypes>;

template <typename T>
struct TypedContainsTest : public ContainsTest {
};

TYPED_TEST_CASE(TypedContainsTest, ContainsTestTypes);

namespace {
template <typename T, std::enable_if_t<cudf::is_numeric<T>(), void>* = nullptr>
auto create_scalar_search_key(T const& value)
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_numeric_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(true);
  static_cast<scalar_type_t<T>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, void>* = nullptr>
auto create_scalar_search_key(std::string const& value)
{
  using namespace cudf;
  using namespace cudf::test;

  return make_string_scalar(value);
}

template <typename T, std::enable_if_t<cudf::is_timestamp<T>(), void>* = nullptr>
auto create_scalar_search_key(typename T::rep const& value)
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_timestamp_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
auto create_scalar_search_key(typename T::rep const& value)
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_duration_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_numeric<T>(), void>* = nullptr>
auto create_null_search_key()
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_numeric_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(false);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_timestamp<T>(), void>* = nullptr>
auto create_null_search_key()
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_timestamp_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(false);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
auto create_null_search_key()
{
  using namespace cudf;
  using namespace cudf::test;

  auto search_key = make_duration_scalar(data_type{type_to_id<T>()});
  search_key->set_valid(false);
  return search_key;
}

}  // namespace

TYPED_TEST(TypedContainsTest, ListContainsScalarWithNoNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1},
    {2, 3, 4},
    {5, 6, 7},
    {8, 9, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key_one = create_scalar_search_key<T>(1);

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 0, 0, 0, 0, 1, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedContainsTest, ListContainsScalarWithNullLists)
{
  // Test List columns that have NULL list rows.

  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 3},
     {}},
    make_counting_transform_iterator(0, [](auto i) {
      return (i != 3) && (i != 10);
    })}.release();

  auto search_key_one = create_scalar_search_key<T>(1);

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result = fixed_width_column_wrapper<bool>{
    {1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
    make_counting_transform_iterator(0, [](auto i) { return (i != 3) && (i != 10); })};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedContainsTest, ListContainsScalarNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto search_space =
    make_lists_column(8,
                      fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                      numerals.release(),
                      0,
                      {});

  auto search_key_one = create_scalar_search_key<T>(1);

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 1, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedContainsTest, ListContainsScalarWithNullsInLists)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>(1);

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, BoolListContainsScalarWithNullsInLists)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = bool;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>(1);

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 1, 0, 0, 1, 0, 1}, {0, 1, 1, 1, 0, 1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, StringListContainsScalarWithNullsInLists)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = std::string;

  auto strings = strings_column_wrapper{
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "1", "2", "3", "4"},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    strings.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_key_one = create_scalar_search_key<T>("1");

  auto actual_result = lists::contains(search_space->view(), *search_key_one);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedContainsTest, ContainsScalarNullSearchKey)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 3},
     {}},
    make_counting_transform_iterator(0, [](auto i) {
      return (i != 3) && (i != 10);
    })}.release();

  auto search_key_null = create_null_search_key<T>();

  auto actual_result = lists::contains(search_space->view(), *search_key_null);

  auto expected_result = fixed_width_column_wrapper<bool>{
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    make_counting_transform_iterator(0, [](auto i) { return false; })};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, ScalarTypeRelatedExceptions)
{
  using namespace cudf;
  using namespace cudf::test;

  {
    // Nested types unsupported.
    auto list_of_lists = lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = create_scalar_search_key<int32_t>(10);
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_lists->view(), *skey),
                              "Nested types not supported in lists::contains()");
  }

  {
    // Search key must match list elements in type.
    auto list_of_ints =
      lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = create_scalar_search_key<std::string>("Hello, World!");
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), *skey),
                              "Type of search key does not match list column element type.");
  }
}

template <typename T>
struct TypedVectorContainsTest : public ContainsTest {
};

using VectorContainsTestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_CASE(TypedVectorContainsTest, VectorContainsTestTypes);

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorWithNoNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1},
    {2, 3, 4},
    {5, 6, 7},
    {8, 9, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1};

  auto actual_result = lists::contains(search_space->view(), search_key);

  auto expected_result = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorWithNullLists)
{
  // Test List columns that have NULL list rows.

  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto search_space = lists_column_wrapper<T, int32_t>{
    {{0, 1, 2},
     {3, 4, 5},
     {6, 7, 8},
     {},
     {9, 0, 1},
     {2, 3, 4},
     {5, 6, 7},
     {8, 9, 0},
     {},
     {1, 2, 3},
     {}},
    make_counting_transform_iterator(0, [](auto i) {
      return (i != 3) && (i != 10);
    })}.release();

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result = fixed_width_column_wrapper<bool>{
    {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    make_counting_transform_iterator(0, [](auto i) { return (i != 3) && (i != 10); })};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto search_space =
    make_lists_column(8,
                      fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
                      numerals.release(),
                      0,
                      {});

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 3};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 1, 1}, {0, 1, 0, 1, 1, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorWithNullsInLists)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 3};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto numerals = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{
    {1, 2, 3, 1, 2, 3, 1, 3}, make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, BoolListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = bool;

  auto numerals = fixed_width_column_wrapper<T, int32_t>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys = fixed_width_column_wrapper<T, int32_t>{
    {0, 1, 0, 1, 0, 0, 1, 1}, make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, StringListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using namespace cudf;
  using namespace cudf::test;

  auto numerals = strings_column_wrapper{
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "1", "2", "3", "4"},
    make_counting_transform_iterator(0, [](auto i) -> bool { return i % 3; })};

  auto input_null_mask_iter = make_counting_transform_iterator(0, [](auto i) { return i != 4; });

  auto search_space = make_lists_column(
    8,
    fixed_width_column_wrapper<size_type>{0, 1, 3, 7, 7, 7, 10, 12, 15}.release(),
    numerals.release(),
    1,
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8));

  auto search_keys =
    strings_column_wrapper{{"1", "2", "3", "1", "2", "3", "1", "3"},
                           make_counting_transform_iterator(0, [](auto i) { return i != 6; })};

  auto actual_result = lists::contains(search_space->view(), search_keys);

  auto expected_result =
    fixed_width_column_wrapper<bool>{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, *actual_result);
}

TEST_F(ContainsTest, VectorTypeRelatedExceptions)
{
  using namespace cudf;
  using namespace cudf::test;

  {
    // Nested types unsupported.
    auto list_of_lists = lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = fixed_width_column_wrapper<int32_t>{0, 1, 2};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_lists->view(), skey),
                              "Nested types not supported in lists::contains()");
  }

  {
    // Search key must match list elements in type.
    auto list_of_ints =
      lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = strings_column_wrapper{"Hello", "World"};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), skey),
                              "Type of search key does not match list column element type.");
  }

  {
    // Search key column size must match lists column size.
    auto list_of_ints = lists_column_wrapper<int32_t>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}.release();

    auto skey = fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
    CUDF_EXPECT_THROW_MESSAGE(lists::contains(list_of_ints->view(), skey),
                              "Number of search keys must match list column size.");
  }
}
