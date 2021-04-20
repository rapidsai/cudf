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
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <unordered_set>

using int_type   = int32_t;
using float_type = float;

using LIST_COL_FLT = cudf::test::lists_column_wrapper<float_type>;
using LIST_COL_STR = cudf::test::lists_column_wrapper<cudf::string_view>;

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
  // -0.0 and 0.0 should be considered equal
  test_once(LIST_COL_FLT{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0},
            LIST_COL_FLT{0, 1, 2});
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsWithInf)
{
  // Lists contain inf
  test_once(LIST_COL_FLT{0, 1, 2, 0, 1, 2, 0, 1, 2, Inf, Inf, Inf}, LIST_COL_FLT{0, 1, 2, Inf});
  test_once(LIST_COL_FLT{Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf},
            LIST_COL_FLT{neg_Inf, 0, Inf});
}

// The position of NaN is undefined after sorting, thus we need to offload the data to CPU to
// check for validity
// We will not store NaN in the results_expected variable (an unordered_set) because we can't check
// for NaN existence in a set. Instead, we will count the number of NaNs in the input and compare
// with the number of NaNs in the output.
static void test_floating_point(std::vector<float_type> const& h_input,
                                std::unordered_set<float_type> const& results_expected,
                                cudf::nan_equality nans_equal)
{
  // If NaNs are considered as equal value, the final result should always contain at max ONE NaN
  // entry per list
  std::size_t const num_NaNs =
    nans_equal == cudf::nan_equality::ALL_EQUAL
      ? std::size_t{1}
      : std::count_if(h_input.begin(), h_input.end(), [](auto x) { return std::isnan(x); });

  auto const results_col = cudf::lists::drop_list_duplicates(
    cudf::lists_column_view{LIST_COL_FLT(h_input.begin(), h_input.end())},
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
  // Trivial cases
  test_once(LIST_COL_STR{{}}, LIST_COL_STR{{}});
  test_once(LIST_COL_STR{"this", "is", "a", "string"}, LIST_COL_STR{"a", "is", "string", "this"});

  // One list column
  test_once(LIST_COL_STR{"this", "is", "is", "is", "a", "string", "string"},
            LIST_COL_STR{"a", "is", "string", "this"});

  // Multiple lists column
  test_once(
    LIST_COL_STR{LIST_COL_STR{"this", "is", "a", "no duplicate", "string"},
                 LIST_COL_STR{"this", "is", "is", "a", "one duplicate", "string"},
                 LIST_COL_STR{"this", "is", "is", "is", "a", "two duplicates", "string"},
                 LIST_COL_STR{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}},
    LIST_COL_STR{LIST_COL_STR{"a", "is", "no duplicate", "string", "this"},
                 LIST_COL_STR{"a", "is", "one duplicate", "string", "this"},
                 LIST_COL_STR{"a", "is", "string", "this", "two duplicates"},
                 LIST_COL_STR{"a", "is", "string", "this", "three duplicates"}});
}

TEST_F(DropListDuplicatesTest, StringTestsWithNulls)
{
  auto const null = std::string("");

  // One list column with null entries
  test_once(
    LIST_COL_STR{{"this", null, "is", "is", "is", "a", null, "string", null, "string"},
                 cudf::detail::make_counting_transform_iterator(
                   0, [](auto i) { return i != 1 && i != 6 && i != 8; })},
    LIST_COL_STR{{"a", "is", "string", "this", null},
                 cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })});

  // Multiple lists column with null lists and null entries
  test_once(
    LIST_COL_STR{
      {LIST_COL_STR{
         {"this", null, "is", null, "a", null, "no duplicate", null, "string"},
         cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; })},
       LIST_COL_STR{},
       LIST_COL_STR{"this", "is", "is", "a", "one duplicate", "string"}},
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })},
    LIST_COL_STR{{LIST_COL_STR{{"a", "is", "no duplicate", "string", "this", null},
                               cudf::detail::make_counting_transform_iterator(
                                 0, [](auto i) { return i <= 4; })},
                  LIST_COL_STR{},
                  LIST_COL_STR{"a", "is", "one duplicate", "string", "this"}},
                 cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })});
}

template <typename T>
struct DropListDuplicatesTypedTest : public cudf::test::BaseFixture {
};
#define LIST_COL cudf::test::lists_column_wrapper<TypeParam>

using TypesForTest =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_CASE(DropListDuplicatesTypedTest, TypesForTest);

TYPED_TEST(DropListDuplicatesTypedTest, InvalidInputTests)
{
  // Lists of nested types are not supported
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{LIST_COL{LIST_COL{{1, 2}, {3}}}}),
    cudf::logic_error);
}

TYPED_TEST(DropListDuplicatesTypedTest, TrivialInputTests)
{
  // Empty input
  test_once(LIST_COL{{}}, LIST_COL{{}});

  // Trivial cases
  test_once(LIST_COL{0, 1, 2, 3, 4, 5}, LIST_COL{0, 1, 2, 3, 4, 5});

  // Multiple empty lists
  test_once(LIST_COL{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}},
            LIST_COL{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}});
}

TYPED_TEST(DropListDuplicatesTypedTest, NonNullInputTests)
{
  // Adjacent lists containing the same entries
  test_once(LIST_COL{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3}},
            LIST_COL{{1}, {1, 2}, {2, 3}});

  // Sliced list column
  auto const list0 =
    LIST_COL{{1, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};
  auto const list1 = cudf::slice(list0, {0, 5})[0];
  auto const list2 = cudf::slice(list0, {1, 5})[0];
  auto const list3 = cudf::slice(list0, {1, 3})[0];
  auto const list4 = cudf::slice(list0, {0, 3})[0];

  test_once(list0, LIST_COL{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once(list1, LIST_COL{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once(list2, LIST_COL{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once(list3, LIST_COL{{1, 2, 3, 4}, {5}});
  test_once(list4, LIST_COL{{1, 2, 3}, {1, 2, 3, 4}, {5}});
}

TYPED_TEST(DropListDuplicatesTypedTest, WithNullInputTests)
{
  auto constexpr null = TypeParam{0};

  // null lists
  test_once(LIST_COL{{{3, 2, 1, 4, 1}, {5}, {}, {}, {10, 8, 9}, {6, 7}},
                     cudf::detail::make_counting_transform_iterator(
                       0, [](auto i) { return i != 2 && i != 3; })},
            LIST_COL{{{1, 2, 3, 4}, {5}, {}, {}, {8, 9, 10}, {6, 7}},
                     cudf::detail::make_counting_transform_iterator(
                       0, [](auto i) { return i != 2 && i != 3; })});

  // null entries are equal
  test_once(
    LIST_COL{std::initializer_list<TypeParam>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
             cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    LIST_COL{std::initializer_list<TypeParam>{1, 3, 5, 7, 9, null},
             cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 5; })});

  // nulls entries are not equal
  test_once(
    LIST_COL{std::initializer_list<TypeParam>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
             cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    LIST_COL{std::initializer_list<TypeParam>{1, 3, 5, 7, 9, null, null, null, null, null},
             cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 5; })},
    cudf::null_equality::UNEQUAL);
}
