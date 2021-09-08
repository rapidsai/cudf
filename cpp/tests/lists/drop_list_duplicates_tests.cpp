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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/drop_list_duplicates.hpp>

#include <algorithm>
#include <unordered_set>

using namespace cudf::test::iterators;

using float_type    = float;
using FloatListsCol = cudf::test::lists_column_wrapper<float_type>;
using StrListsCol   = cudf::test::lists_column_wrapper<cudf::string_view>;

auto constexpr neg_NaN = -std::numeric_limits<float_type>::quiet_NaN();
auto constexpr neg_Inf = -std::numeric_limits<float_type>::infinity();
auto constexpr NaN     = std::numeric_limits<float_type>::quiet_NaN();
auto constexpr Inf     = std::numeric_limits<float_type>::infinity();

template <class LCW>
void test_once(cudf::column_view const& input,
               LCW const& expected,
               cudf::null_equality nulls_equal = cudf::null_equality::EQUAL)
{
  auto const results =
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{input}, nulls_equal);
  if (cudf::is_floating_point(input.type())) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  } else {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

struct DropListDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithSignedZero)
{
  // -0.0 and 0.0 should be considered equal.
  auto const lists    = FloatListsCol{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0};
  auto const expected = FloatListsCol{0, 1, 2};
  auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithInf)
{
  // Lists contain inf.
  {
    auto const lists    = FloatListsCol{0, 1, 2, 0, 1, 2, 0, 1, 2, Inf, Inf, Inf};
    auto const expected = FloatListsCol{0, 1, 2, Inf};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {
    auto const lists    = FloatListsCol{Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf};
    auto const expected = FloatListsCol{neg_Inf, 0, Inf};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

// The position of NaN is undefined after sorting, thus we need to offload the data to CPU to
// check for validity.
// We will not store NaN in the results_expected variable (an unordered_set) because we can't check
// for NaN existence in a set. Instead, we will count the number of NaNs in the input and compare
// with the number of NaNs in the output.
static void test_floating_point(std::vector<float_type> const& h_input,
                                std::unordered_set<float_type> const& results_expected,
                                cudf::nan_equality nans_equal)
{
  // If NaNs are considered as equal value, the final result should always contain at max ONE NaN
  // entry per list.
  std::size_t const num_NaNs =
    nans_equal == cudf::nan_equality::ALL_EQUAL
      ? std::size_t{1}
      : std::count_if(h_input.begin(), h_input.end(), [](auto x) { return std::isnan(x); });

  auto const results_col = cudf::lists::drop_list_duplicates(
    cudf::lists_column_view{FloatListsCol(h_input.begin(), h_input.end())},
    cudf::null_equality::EQUAL,
    nans_equal);
  auto const results_arr =
    cudf::test::to_host<float_type>(cudf::lists_column_view(results_col->view()).child()).first;

  EXPECT_EQ(results_arr.size(), results_expected.size() + num_NaNs);

  std::size_t NaN_count{0};
  std::unordered_set<float_type> results;
  for (auto const x : results_arr) {
    if (std::isnan(x)) {
      ++NaN_count;
    } else {
      results.insert(x);
    }
  }
  EXPECT_TRUE(results_expected.size() == results.size() && NaN_count == num_NaNs);
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithNaNs)
{
  std::vector<float_type> h_input{
    0, -1, 1, NaN, 2, 0, neg_NaN, 1, -2, 2, 0, 1, 2, neg_NaN, NaN, NaN, NaN, neg_NaN};
  std::unordered_set<float_type> results_expected{-2, -1, 0, 1, 2};
  test_floating_point(h_input, results_expected, cudf::nan_equality::UNEQUAL);
  test_floating_point(h_input, results_expected, cudf::nan_equality::ALL_EQUAL);
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithInfsAndNaNs)
{
  std::vector<float_type> h_input{neg_Inf, 0, neg_NaN, 1,   -1,      -2,      NaN, NaN,    Inf, NaN,
                                  neg_NaN, 2, -1,      0,   neg_NaN, 1,       2,   Inf,    0,   1,
                                  neg_Inf, 2, neg_NaN, Inf, neg_NaN, neg_NaN, NaN, neg_Inf};
  std::unordered_set<float_type> results_expected{-2, -1, 0, 1, 2, neg_Inf, Inf};
  test_floating_point(h_input, results_expected, cudf::nan_equality::UNEQUAL);
  test_floating_point(h_input, results_expected, cudf::nan_equality::ALL_EQUAL);
}

TEST_F(DropListDuplicatesTest, StringTestsNonNull)
{
  // Trivial cases - empty input.
  {
    auto const lists    = StrListsCol{{}};
    auto const expected = StrListsCol{{}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {
    auto const lists    = StrListsCol{"this", "is", "a", "string"};
    auto const expected = StrListsCol{"a", "is", "string", "this"};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // One list column.
  {
    auto const lists    = StrListsCol{"this", "is", "is", "is", "a", "string", "string"};
    auto const expected = StrListsCol{"a", "is", "string", "this"};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // Multiple lists column.
  {
    auto const lists =
      StrListsCol{StrListsCol{"this", "is", "a", "no duplicate", "string"},
                  StrListsCol{"this", "is", "is", "a", "one duplicate", "string"},
                  StrListsCol{"this", "is", "is", "is", "a", "two duplicates", "string"},
                  StrListsCol{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}};
    auto const expected = StrListsCol{StrListsCol{"a", "is", "no duplicate", "string", "this"},
                                      StrListsCol{"a", "is", "one duplicate", "string", "this"},
                                      StrListsCol{"a", "is", "string", "this", "two duplicates"},
                                      StrListsCol{"a", "is", "string", "this", "three duplicates"}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TEST_F(DropListDuplicatesTest, StringTestsWithNulls)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const lists = StrListsCol{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const expected = StrListsCol{{"a", "is", "string", "this", null}, null_at(4)};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // Multiple lists column with null lists and null entries
  {
    auto const lists = StrListsCol{
      {StrListsCol{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                   nulls_at({1, 3, 5, 7})},
       StrListsCol{}, /* NULL */
       StrListsCol{"this", "is", "is", "a", "one duplicate", "string"}},
      null_at(1)};
    auto const expected =
      StrListsCol{{StrListsCol{{"a", "is", "no duplicate", "string", "this", null}, null_at(5)},
                   StrListsCol{}, /* NULL */
                   StrListsCol{"a", "is", "one duplicate", "string", "this"}},
                  null_at(1)};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

template <typename T>
struct DropListDuplicatesTypedTest : public cudf::test::BaseFixture {
};

using TypesForTest =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(DropListDuplicatesTypedTest, TypesForTest);

TYPED_TEST(DropListDuplicatesTypedTest, InvalidInputTests)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  // Nested types (except struct) are not supported.
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{ListsCol{ListsCol{{1, 2}, {3}}}}),
    cudf::logic_error);
}

TYPED_TEST(DropListDuplicatesTypedTest, TrivialInputTests)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  // Empty input.
  {
    auto const lists    = ListsCol{{}};
    auto const expected = ListsCol{{}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // Trivial cases.
  {
    auto const lists    = ListsCol{0, 1, 2, 3, 4, 5};
    auto const expected = ListsCol{0, 1, 2, 3, 4, 5};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // Multiple empty lists.
  {
    auto const lists    = ListsCol{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}};
    auto const expected = ListsCol{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, NonNullInputTests)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  // Adjacent lists containing the same entries.
  {
    auto const lists =
      ListsCol{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3}};
    auto const expected = ListsCol{{1}, {1, 2}, {2, 3}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // Sliced list column.
  auto const lists_original =
    ListsCol{{1, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};
  auto const lists1 = cudf::slice(lists_original, {0, 5})[0];
  auto const lists2 = cudf::slice(lists_original, {1, 5})[0];
  auto const lists3 = cudf::slice(lists_original, {1, 3})[0];
  auto const lists4 = cudf::slice(lists_original, {0, 3})[0];

  {
    auto const expected = ListsCol{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists_original});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  {
    auto const expected = ListsCol{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  {
    auto const expected = ListsCol{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  {
    auto const expected = ListsCol{{1, 2, 3, 4}, {5}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  {
    auto const expected = ListsCol{{1, 2, 3}, {1, 2, 3, 4}, {5}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, WithNullInputTests)
{
  using ListsCol      = cudf::test::lists_column_wrapper<TypeParam>;
  auto constexpr null = TypeParam{0};

  // null lists.
  {
    auto const lists = ListsCol{
      {{3, 2, 1, 4, 1}, {5}, {} /*NULL*/, {} /*NULL*/, {10, 8, 9}, {6, 7}}, nulls_at({2, 3})};
    auto const expected =
      ListsCol{{{1, 2, 3, 4}, {5}, {} /*NULL*/, {} /*NULL*/, {8, 9, 10}, {6, 7}}, nulls_at({2, 3})};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // null entries are equal.
  {
    auto const lists = ListsCol{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const expected =
      ListsCol{std::initializer_list<TypeParam>{1, 3, 5, 7, 9, null}, null_at(5)};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  // nulls entries are not equal.
  {
    auto const lists = ListsCol{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const expected =
      ListsCol{std::initializer_list<TypeParam>{1, 3, 5, 7, 9, null, null, null, null, null},
               nulls_at({5, 6, 7, 8, 9})};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists},
                                                           cudf::null_equality::UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}
