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

#include <cudf/copying.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/drop_list_duplicates.hpp>

#include <algorithm>
#include <unordered_set>

using namespace cudf::test::iterators;

using float_type    = float;
using IntListsCol   = cudf::test::lists_column_wrapper<int32_t>;
using FloatListsCol = cudf::test::lists_column_wrapper<float_type>;
using StrListsCol   = cudf::test::lists_column_wrapper<cudf::string_view>;
using StringsCol    = cudf::test::strings_column_wrapper;
using StructsCol    = cudf::test::structs_column_wrapper;
using IntsCol       = cudf::test::fixed_width_column_wrapper<int32_t>;
using FloatsCol     = cudf::test::fixed_width_column_wrapper<float_type>;

auto constexpr neg_NaN   = -std::numeric_limits<float_type>::quiet_NaN();
auto constexpr neg_Inf   = -std::numeric_limits<float_type>::infinity();
auto constexpr NaN       = std::numeric_limits<float_type>::quiet_NaN();
auto constexpr Inf       = std::numeric_limits<float_type>::infinity();
auto constexpr verbosity = cudf::test::debug_output_level::FIRST_ERROR;

struct DropListDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithSignedZero)
{
  // -0.0 and 0.0 should be considered equal.
  auto const keys = FloatListsCol{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0, 3};
  auto const vals =
    StrListsCol{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"};
  auto const expected_keys = FloatListsCol{0, 1, 2, 3};

  // Remove duplicates only from keys.
  {
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_keys, verbosity);
  }

  // Remove duplicates with KEEP_FIRST.
  {
    auto const expected_vals = StrListsCol{"1", "2", "3", "14"};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_FIRST);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }

  // Remove duplicates with KEEP_LAST.
  {
    auto const expected_vals = StrListsCol{"13", "8", "9", "14"};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_LAST);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }

  // Remove duplicates with KEEP_NONE.
  {
    auto const expected_keys = FloatListsCol{3};
    auto const expected_vals = StrListsCol{"14"};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_NONE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithInf)
{
  auto const keys          = FloatListsCol{Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf};
  auto const vals          = IntListsCol{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto const expected_keys = FloatListsCol{neg_Inf, 0, Inf};

  // Remove duplicates only from keys.
  {
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_keys, verbosity);
  }

  // Remove duplicates with KEEP_FIRST.
  {
    auto const expected_vals = IntListsCol{3, 2, 1};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_FIRST);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }

  // Remove duplicates with KEEP_LAST.
  {
    auto const expected_vals = IntListsCol{11, 10, 9};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_LAST);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }

  // Remove duplicates with KEEP_NONE.
  {
    auto const expected_keys = FloatListsCol{FloatListsCol{}};
    auto const expected_vals = IntListsCol{IntListsCol{}};
    auto const [results_keys, results_vals] =
      cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                        cudf::lists_column_view{vals},
                                        cudf::duplicate_keep_option::KEEP_NONE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
  }
  //  exit(0);
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  // No duplicate entry.
  {
    auto const lists    = StrListsCol{"this", "is", "a", "string"};
    auto const expected = StrListsCol{"a", "is", "string", "this"};
    auto const results  = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  // One list column.
  {
    auto const lists    = StrListsCol{"this", "is", "is", "is", "a", "string", "string"};
    auto const expected = StrListsCol{"a", "is", "string", "this"};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  // One list column, input is a strings column with given non-default null_equality and
  // nans_equality parameters.
  {
    auto const lists    = StrListsCol{"this", "is", "is", "is", "a", "string", "string"};
    auto const expected = StrListsCol{"a", "is", "string", "this"};
    auto const results  = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
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
    auto const lists    = ListsCol{};
    auto const expected = ListsCol{};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);

    auto const [results_keys, results_vals] = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected, verbosity);
  }

  // All input lists are empty.
  {
    auto const lists    = ListsCol{ListsCol{}, ListsCol{}, ListsCol{}};
    auto const expected = ListsCol{ListsCol{}, ListsCol{}, ListsCol{}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);

    auto const [results_keys, results_vals] = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected, verbosity);
  }

  // Trivial cases.
  {
    auto const lists    = ListsCol{0, 1, 2, 3, 4, 5};
    auto const expected = ListsCol{0, 1, 2, 3, 4, 5};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);

    auto const [results_keys, results_vals] = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected, verbosity);
  }

  // Multiple empty lists.
  {
    auto const lists    = ListsCol{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}};
    auto const expected = ListsCol{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);

    auto const [results_keys, results_vals] = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists}, cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected, verbosity);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, NonNullInputTests)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  // Adjacent lists containing the same entries.
  {
    auto const keys =
      ListsCol{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3}};
    auto const vals =
      ListsCol{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    auto const expected_keys = ListsCol{{1}, {1, 2}, {2, 3}};

    // Remove duplicates with KEEP_FIRST.
    {
      auto const expected_vals = ListsCol{{1}, {1, 6}, {1, 5}};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_FIRST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_LAST.
    {
      auto const expected_vals = ListsCol{{8}, {5, 8}, {4, 8}};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_LAST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_NONE.
    {
      auto const expected = ListsCol{ListsCol{}, ListsCol{}, ListsCol{}};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_NONE);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected, verbosity);
    }
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  {
    auto const expected = ListsCol{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  {
    auto const expected = ListsCol{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  {
    auto const expected = ListsCol{{1, 2, 3, 4}, {5}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }

  {
    auto const expected = ListsCol{{1, 2, 3}, {1, 2, 3, 4}, {5}};
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, WithNullInputTests)
{
  using ListsCol      = cudf::test::lists_column_wrapper<TypeParam>;
  auto constexpr null = TypeParam{0};

  // null entries and lists.
  {
    auto const keys = ListsCol{{{3, 2, 1, 4, 1}, {5}, {} /*NULL*/, {} /*NULL*/, {10, 8, 9}, {6, 7}},
                               nulls_at({2, 3})};
    auto const vals =
      ListsCol{{ListsCol{{1, 2, null, 4, 5}, null_at(2)}, {1}, {}, {} /*NULL*/, {1, 2, 3}, {1, 2}},
               null_at(3)};
    auto const expected_keys =
      ListsCol{{{1, 2, 3, 4}, {5}, {} /*NULL*/, {} /*NULL*/, {8, 9, 10}, {6, 7}}, nulls_at({2, 3})};

    // Remove duplicates with KEEP_FIRST.
    {
      auto const expected_vals =
        ListsCol{{ListsCol{{null, 2, 1, 4}, null_at(0)}, {1}, {}, {} /*NULL*/, {2, 3, 1}, {1, 2}},
                 null_at(3)};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_FIRST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_LAST.
    {
      auto const expected_vals =
        ListsCol{{ListsCol{5, 2, 1, 4}, {1}, {}, {} /*NULL*/, {2, 3, 1}, {1, 2}}, null_at(3)};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_LAST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_NONE.
    {
      auto const expected_keys =
        ListsCol{{{2, 3, 4}, {5}, {} /*NULL*/, {} /*NULL*/, {8, 9, 10}, {6, 7}}, nulls_at({2, 3})};
      auto const expected_vals =
        ListsCol{{ListsCol{2, 1, 4}, {1}, {}, {} /*NULL*/, {2, 3, 1}, {1, 2}}, null_at(3)};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_NONE);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results_vals->view(), expected_vals, verbosity);
    }
  }

  // null entries are equal.
  {
    auto const keys =
      ListsCol{{null, 1, null, 3, null, 5, null, 7, null, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const vals = ListsCol{{null, 1, 2, 3, 4, null, 6, 7, 8, null}, nulls_at({0, 5, 9})};
    auto const expected_keys = ListsCol{{1, 3, 5, 7, 9, null}, null_at(5)};

    // Remove duplicates with KEEP_FIRST.
    {
      auto const expected_vals = ListsCol{{1, 3, null, 7, null, null}, nulls_at({2, 4, 5})};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_FIRST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_LAST.
    {
      auto const expected_vals = ListsCol{{1, 3, null, 7, null, 8}, nulls_at({2, 4})};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_LAST);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }

    // Remove duplicates with KEEP_NONE.
    {
      auto const expected_keys = ListsCol{1, 3, 5, 7, 9};
      auto const expected_vals = ListsCol{{1, 3, null, 7, null}, nulls_at({2, 4})};
      auto const [results_keys, results_vals] =
        cudf::lists::drop_list_duplicates(cudf::lists_column_view{keys},
                                          cudf::lists_column_view{vals},
                                          cudf::duplicate_keep_option::KEEP_NONE);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results_keys->view(), expected_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_vals->view(), expected_vals, verbosity);
    }
  }

  // null entries are not equal.
  {
    auto const lists = ListsCol{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const expected =
      ListsCol{std::initializer_list<TypeParam>{1, 3, 5, 7, 9, null, null, null, null, null},
               nulls_at({5, 6, 7, 8, 9})};
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists},
                                                           cudf::null_equality::UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, InputListsOfStructsNoNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const get_structs = [] {
    auto child1 = ColWrapper{
      1, 1, 1, 1, 1, 1, 1, 1,  // list1
      1, 1, 1, 1, 2, 1, 2, 2,  // list2
      2, 2, 2, 2, 3, 2, 3, 3   // list3
    };
    auto child2 = StringsCol{
      // begin list1
      "Banana",
      "Mango",
      "Apple",
      "Cherry",
      "Kiwi",
      "Banana",
      "Cherry",
      "Kiwi",  // end list1
      // begin list2
      "Bear",
      "Duck",
      "Cat",
      "Dog",
      "Panda",
      "Bear",
      "Cat",
      "Panda",  // end list2
      // begin list3
      "ÁÁÁ",
      "ÉÉÉÉÉ",
      "ÍÍÍÍÍ",
      "ÁBC",
      "XYZ",
      "ÁÁÁ",
      "ÁBC",
      "XYZ"  // end list3
    };
    return StructsCol{{child1, child2}};
  };

  auto const get_structs_expected = [] {
    auto child1 = ColWrapper{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3};
    auto child2 = StringsCol{
      // begin list1
      "Apple",
      "Banana",
      "Cherry",
      "Kiwi",
      "Mango",  // end list1
      // begin list2
      "Bear",
      "Cat",
      "Dog",
      "Duck",
      "Cat",
      "Panda",  // end list2
      // begin list3
      "ÁBC",
      "ÁÁÁ",
      "ÉÉÉÉÉ",
      "ÍÍÍÍÍ",
      "XYZ",
      "ÁBC"  // end list3
    };
    return StructsCol{{child1, child2}};
  };

  // Test full columns.
  {
    auto const lists =
      cudf::make_lists_column(3, IntsCol{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, IntsCol{0, 5, 11, 17}.release(), get_structs_expected().release(), 0, {});
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists->view()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected->view(), verbosity);
  }

  // Test sliced columns.
  {
    auto const lists_original =
      cudf::make_lists_column(3, IntsCol{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected_original = cudf::make_lists_column(
      3, IntsCol{0, 5, 11, 17}.release(), get_structs_expected().release(), 0, {});
    auto const lists    = cudf::slice(lists_original->view(), {1, 3})[0];
    auto const expected = cudf::slice(expected_original->view(), {1, 3})[0];
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, InputListsOfStructsHaveNull)
{
  using ColWrapper    = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  auto constexpr XXX  = int32_t{0};  // nulls at the parent structs column level
  auto constexpr null = int32_t{0};  // nulls at the children columns level

  auto const get_structs = [] {
    auto child1 = ColWrapper{{
                               1,    1,    null, XXX, XXX, 1, 1,    1,  // list1
                               1,    1,    1,    1,   2,   1, null, 2,  // list2
                               null, null, 2,    2,   3,   2, 3,    3   // list3
                             },
                             nulls_at({2, 14, 16, 17})};
    auto child2 = StringsCol{{
                               // begin list1
                               "Banana",
                               "Mango",
                               "Apple",
                               "XXX", /*NULL*/
                               "XXX", /*NULL*/
                               "Banana",
                               "Cherry",
                               "Kiwi",  // end list1
                                        // begin list2
                               "Bear",
                               "Duck",
                               "Cat",
                               "Dog",
                               "Panda",
                               "Bear",
                               "" /*NULL*/,
                               "Panda",  // end list2
                                         // begin list3
                               "ÁÁÁ",
                               "ÉÉÉÉÉ",
                               "ÍÍÍÍÍ",
                               "ÁBC",
                               "" /*NULL*/,
                               "ÁÁÁ",
                               "ÁBC",
                               "XYZ"  // end list3
                             },
                             nulls_at({14, 20})};
    return StructsCol{{child1, child2}, nulls_at({3, 4})};
  };

  auto const get_structs_expected = [] {
    auto child1 =
      ColWrapper{{1, 1, 1, 1, null, XXX, 1, 1, 1, 1, 2, null, 2, 2, 2, 3, 3, 3, null, null},
                 nulls_at({4, 5, 11, 18, 19})};
    auto child2 = StringsCol{{
                               // begin list1
                               "Banana",
                               "Cherry",
                               "Kiwi",
                               "Mango",
                               "Apple",
                               "XXX" /*NULL*/,  // end list1
                                                // begin list2
                               "Bear",
                               "Cat",
                               "Dog",
                               "Duck",
                               "Panda",
                               "" /*NULL*/,  // end list2
                                             // begin list3
                               "ÁBC",
                               "ÁÁÁ",
                               "ÍÍÍÍÍ",
                               "XYZ",
                               "ÁBC",
                               "" /*NULL*/,
                               "ÁÁÁ",
                               "ÉÉÉÉÉ"  // end list3
                             },
                             nulls_at({5, 11, 17})};
    return StructsCol{{child1, child2}, null_at(5)};
  };

  // Test full columns.
  {
    auto const lists =
      cudf::make_lists_column(3, IntsCol{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, IntsCol{0, 6, 12, 20}.release(), get_structs_expected().release(), 0, {});
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists->view()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected->view(), verbosity);
  }

  // Test sliced columns.
  {
    auto const lists_original =
      cudf::make_lists_column(3, IntsCol{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected_original = cudf::make_lists_column(
      3, IntsCol{0, 6, 12, 20}.release(), get_structs_expected().release(), 0, {});
    auto const lists    = cudf::slice(lists_original->view(), {1, 3})[0];
    auto const expected = cudf::slice(expected_original->view(), {1, 3})[0];
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }
}

TYPED_TEST(DropListDuplicatesTypedTest, InputListsOfNestedStructsHaveNull)
{
  using ColWrapper    = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  auto constexpr null = int32_t{0};  // nulls at the children columns level
  // XXX and YYY are int placeholders for nulls at parent structs column level.
  // We bring up two placeholders of different values to create intra null structs with
  // children of different values, so as to test whether null_equality::EQUAL works or not.
  auto constexpr XXX = int32_t{5};
  auto constexpr YYY = int32_t{6};

  auto const get_nested_structs = [] {
    auto grandchild1 = ColWrapper{{
                                    1,    XXX,  null, XXX, YYY, 1, 1,    1,  // list1
                                    1,    1,    1,    1,   2,   1, null, 2,  // list2
                                    null, null, 2,    2,   3,   2, 3,    3   // list3
                                  },
                                  nulls_at({2, 14, 16, 17})};
    auto grandchild2 = StringsCol{{
                                    // begin list1
                                    "Banana",
                                    "YYY", /*NULL*/
                                    "Apple",
                                    "XXX", /*NULL*/
                                    "YYY", /*NULL*/
                                    "Banana",
                                    "Cherry",
                                    "Kiwi",  // end list1
                                             // begin list2
                                    "Bear",
                                    "Duck",
                                    "Cat",
                                    "Dog",
                                    "Panda",
                                    "Bear",
                                    "" /*NULL*/,
                                    "Panda",  // end list2
                                              // begin list3
                                    "ÁÁÁ",
                                    "ÉÉÉÉÉ",
                                    "ÍÍÍÍÍ",
                                    "ÁBC",
                                    "" /*NULL*/,
                                    "ÁÁÁ",
                                    "ÁBC",
                                    "XYZ"  // end list3
                                  },
                                  nulls_at({14, 20})};
    auto child1      = StructsCol{{grandchild1, grandchild2}, nulls_at({1, 3, 4})};
    return StructsCol{{child1}};
  };

  auto const get_nested_struct_expected = [] {
    auto grandchild1 =
      ColWrapper{{1, 1, 1, null, XXX, 1, 1, 1, 1, 2, null, 2, 2, 2, 3, 3, 3, null, null},
                 nulls_at({3, 4, 10, 17, 18})};
    auto grandchild2 = StringsCol{{
                                    // begin list1
                                    "Banana",
                                    "Cherry",
                                    "Kiwi",
                                    "Apple",
                                    "XXX" /*NULL*/,  // end list1
                                                     // begin list2
                                    "Bear",
                                    "Cat",
                                    "Dog",
                                    "Duck",
                                    "Panda",
                                    "" /*NULL*/,  // end list2
                                                  // begin list3
                                    "ÁBC",
                                    "ÁÁÁ",
                                    "ÍÍÍÍÍ",
                                    "XYZ",
                                    "ÁBC",
                                    "" /*NULL*/,
                                    "ÁÁÁ",
                                    "ÉÉÉÉÉ"  // end list3
                                  },
                                  nulls_at({4, 10, 16})};
    auto child1      = StructsCol{{grandchild1, grandchild2}, nulls_at({4})};
    return StructsCol{{child1}};
  };

  // Test full columns.
  {
    auto const lists = cudf::make_lists_column(
      3, IntsCol{0, 8, 16, 24}.release(), get_nested_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, IntsCol{0, 5, 11, 19}.release(), get_nested_struct_expected().release(), 0, {});
    auto const results = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists->view()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected->view(), verbosity);
  }

  // Test sliced columns.
  {
    auto const lists_original = cudf::make_lists_column(
      3, IntsCol{0, 8, 16, 24}.release(), get_nested_structs().release(), 0, {});
    auto const expected_original = cudf::make_lists_column(
      3, IntsCol{0, 5, 11, 19}.release(), get_nested_struct_expected().release(), 0, {});
    auto const lists    = cudf::slice(lists_original->view(), {1, 3})[0];
    auto const expected = cudf::slice(expected_original->view(), {1, 3})[0];
    auto const results  = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, verbosity);
  }
}

TEST_F(DropListDuplicatesTest, SlicedInputListsOfStructsWithNaNs)
{
  auto const h_child = std::vector<float_type>{
    0, -1, 1, 0, 2, 0, 1, 1, -2, 2, 0, 1, 2, neg_NaN, NaN, NaN, NaN, neg_NaN};

  auto const get_structs = [&] {
    // Two children are just identical.
    auto child1 = FloatsCol(h_child.begin(), h_child.end());
    auto child2 = FloatsCol(h_child.begin(), h_child.end());
    return StructsCol{{child1, child2}};
  };

  // The first list does not have any NaN or -NaN, while the second list has both.
  // `drop_list_duplicates` is expected to operate properly on this second list.
  auto const lists_original =
    cudf::make_lists_column(2, IntsCol{0, 10, 18}.release(), get_structs().release(), 0, {});
  auto const lists2 = cudf::slice(lists_original->view(), {1, 2})[0];  // test on the second list

  // Contain expected vals excluding NaN.
  auto const results_children_expected = std::unordered_set<float_type>{0, 1, 2};

  // Test for cudf::nan_equality::UNEQUAL.
  {
    auto const results_col = cudf::lists::drop_list_duplicates(cudf::lists_column_view{lists2});
    auto const child       = cudf::lists_column_view(results_col->view()).child();
    auto const results_arr = cudf::test::to_host<float_type>(child.child(0)).first;

    std::size_t const num_NaNs =
      std::count_if(h_child.begin(), h_child.end(), [](auto x) { return std::isnan(x); });
    EXPECT_EQ(results_arr.size(), results_children_expected.size() + num_NaNs);

    std::size_t NaN_count{0};
    std::unordered_set<float_type> results;
    for (auto const x : results_arr) {
      if (std::isnan(x)) {
        ++NaN_count;
      } else {
        results.insert(x);
      }
    }
    EXPECT_TRUE(results_children_expected.size() == results.size() && NaN_count == num_NaNs);
  }

  // Test for cudf::nan_equality::ALL_EQUAL.
  {
    auto const results_col = cudf::lists::drop_list_duplicates(
      cudf::lists_column_view{lists2}, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL);
    auto const child       = cudf::lists_column_view(results_col->view()).child();
    auto const results_arr = cudf::test::to_host<float_type>(child.child(0)).first;

    std::size_t const num_NaNs = 1;
    EXPECT_EQ(results_arr.size(), results_children_expected.size() + num_NaNs);

    std::size_t NaN_count{0};
    std::unordered_set<float_type> results;
    for (auto const x : results_arr) {
      if (std::isnan(x)) {
        ++NaN_count;
      } else {
        results.insert(x);
      }
    }
    EXPECT_TRUE(results_children_expected.size() == results.size() && NaN_count == num_NaNs);
  }
}
