/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/set_operations.hpp>

using float_type = double;
using namespace cudf::test::iterators;

auto constexpr null{0};  // null at current level
// auto constexpr XXX{0};   // null pushed down from parent level
auto constexpr neg_NaN      = -std::numeric_limits<float_type>::quiet_NaN();
auto constexpr neg_Inf      = -std::numeric_limits<float_type>::infinity();
auto constexpr NaN          = std::numeric_limits<float_type>::quiet_NaN();
auto constexpr Inf          = std::numeric_limits<float_type>::infinity();
auto constexpr NULL_EQUAL   = cudf::null_equality::EQUAL;
auto constexpr NULL_UNEQUAL = cudf::null_equality::UNEQUAL;
auto constexpr NAN_EQUAL    = cudf::nan_equality::ALL_EQUAL;
auto constexpr NAN_UNEQUAL  = cudf::nan_equality::UNEQUAL;

using bools_col     = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_lists  = cudf::test::lists_column_wrapper<float_type>;
using strings_lists = cudf::test::lists_column_wrapper<cudf::string_view>;
using strings_col   = cudf::test::strings_column_wrapper;
using structs_col   = cudf::test::structs_column_wrapper;
using lists_cv      = cudf::lists_column_view;

// using cudf::nan_policy;
// using cudf::null_equality;
// using cudf::null_policy;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

struct ListOverlapTest : public cudf::test::BaseFixture {
};

template <typename T>
struct ListOverlapTypedTest : public cudf::test::BaseFixture {
};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(ListOverlapTypedTest, TestTypes);

TEST_F(ListOverlapTest, TrivialTest)
{
  auto const lhs =
    floats_lists{{floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 0.0}, null_at(6)},
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)},
                  {} /*NULL*/,
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)}},
                 null_at(2)};
  auto const rhs =
    floats_lists{{floats_lists{{1.0, 0.5, null, 0.0, 0.0, null, NaN}, nulls_at({2, 5})},
                  floats_lists{{2.0, 1.0, null, 0.0, 0.0, null}, nulls_at({2, 5})},
                  floats_lists{{2.0, 1.0, null, 0.0, 0.0, null}, nulls_at({2, 5})},
                  {} /*NULL*/},
                 null_at(3)};

  auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
  auto const expected = bools_col{{1, 1, null, null}, nulls_at({2, 3})};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
}

TEST_F(ListOverlapTest, FloatingPointTestsWithSignedZero)
{
  // -0.0 and 0.0 should be considered equal.
  auto const lhs      = floats_lists{{0.0, 0.0, 0.0, 0.0, 0.0}, {-0.0, 1.0}, {0.0}};
  auto const rhs      = floats_lists{{-0.0, -0.0, -0.0, -0.0, -0.0}, {0.0, 2.0}, {1.0}};
  auto const expected = bools_col{1, 1, 0};
  auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
}

TEST_F(ListOverlapTest, FloatingPointTestsWithInf)
{
  auto const lhs      = floats_lists{{Inf, Inf, Inf}, {Inf, 0.0, neg_Inf}};
  auto const rhs      = floats_lists{{neg_Inf, neg_Inf}, {0.0}};
  auto const expected = bools_col{0, 1};
  auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
}

TEST_F(ListOverlapTest, FloatingPointTestsWithNaNs)
{
  auto const lhs =
    floats_lists{{0, -1, 1, NaN}, {2, 0, neg_NaN}, {1, -2, 2, 0, 1, 2}, {NaN, NaN, NaN, NaN, NaN}};
  auto const rhs =
    floats_lists{{2, 3, 4, neg_NaN}, {2, 0}, {neg_NaN, 1, -2, 2, 0, 1, 2}, {neg_NaN, neg_NaN}};

  // NaNs are equal.
  {
    auto const expected = bools_col{1, 1, 1, 1};
    auto const results =
      cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_EQUAL, NAN_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // NaNs are unequal.
  {
    auto const expected = bools_col{0, 1, 1, 0};
    auto const results =
      cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_EQUAL, NAN_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}

TEST_F(ListOverlapTest, StringTestsNonNull)
{
  // Trivial cases - empty input.
  {
    auto const lhs      = strings_lists{};
    auto const rhs      = strings_lists{};
    auto const expected = bools_col{};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // Trivial cases - empty input.
  {
    auto const lhs      = strings_lists{strings_lists{}};
    auto const rhs      = strings_lists{strings_lists{}};
    auto const expected = bools_col{0};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // No overlap.
  {
    auto const lhs      = strings_lists{"this", "is", "a", "string"};
    auto const rhs      = strings_lists{"aha", "bear", "blow", "heat"};
    auto const expected = bools_col{0};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // One list column.
  {
    auto const lhs      = strings_lists{"this", "is", "a", "string"};
    auto const rhs      = strings_lists{"a", "delicious", "banana"};
    auto const expected = bools_col{1};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // Multiple lists column.
  {
    auto const lhs      = strings_lists{strings_lists{"one", "two", "three"},
                                   strings_lists{"four", "five", "six"},
                                   strings_lists{"1", "2", "3"}};
    auto const rhs      = strings_lists{strings_lists{"one", "banana"},
                                   strings_lists{"apple", "kiwi", "cherry"},
                                   strings_lists{"two", "and", "1"}};
    auto const expected = bools_col{1, 0, 1};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}

TEST_F(ListOverlapTest, StringTestsWithNullsEqual)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const lhs = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const rhs =
      strings_lists{{"aha", null, "abc", null, "1111", null, "2222"}, nulls_at({1, 3, 5})};
    auto const expected = bools_col{1};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const lhs = strings_lists{
      strings_lists{{"this", null, "is", null, "a", null, null, "string"}, nulls_at({1, 3, 5, 6})},
      strings_lists{},
      strings_lists{"this", "is", "a", "string"}};
    auto const rhs = strings_lists{
      {strings_lists{{"aha", null, "abc", null, "1111", null, "2222"}, nulls_at({1, 3, 5})},
       strings_lists{}, /* NULL */
       strings_lists{"aha", "this", "is another", "string???"}},
      null_at(1)};
    auto const expected = bools_col{{1, 0 /*null*/, 1}, null_at(1)};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}

TEST_F(ListOverlapTest, StringTestsWithNullsUnequal)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const lhs = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const rhs =
      strings_lists{{"aha", null, "abc", null, "1111", null, "2222"}, nulls_at({1, 3, 5})};
    auto const expected = bools_col{0};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const lhs = strings_lists{
      strings_lists{{"this", null, "is", null, "a", null, null, "string"}, nulls_at({1, 3, 5, 6})},
      strings_lists{},
      strings_lists{"this", "is", "a", "string"}};
    auto const rhs = strings_lists{
      {strings_lists{{"aha", null, "abc", null, "1111", null, "2222"}, nulls_at({1, 3, 5})},
       strings_lists{}, /* NULL */
       strings_lists{"aha", "this", "is another", "string???"}},
      null_at(1)};
    auto const expected = bools_col{{0, 0 /*null*/, 1}, null_at(1)};
    auto const results  = cudf::lists::list_overlap(lists_cv{lhs}, lists_cv{rhs}, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}
