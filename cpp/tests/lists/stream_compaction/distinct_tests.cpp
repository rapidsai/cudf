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

#include <cudf/lists/sorting.hpp>
#include <cudf/lists/stream_compaction.hpp>

using float_type = double;
using namespace cudf::test::iterators;
// using cudf::nan_policy;
// using cudf::null_equality;
// using cudf::null_policy;

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

using bools_col = cudf::test::fixed_width_column_wrapper<bool>;
// using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col    = cudf::test::fixed_width_column_wrapper<float_type>;
using floats_lists  = cudf::test::lists_column_wrapper<float_type>;
using strings_lists = cudf::test::lists_column_wrapper<cudf::string_view>;
// using strings_col = cudf::test::strings_column_wrapper;
// using structs_col = cudf::test::structs_column_wrapper;
using lists_cv = cudf::lists_column_view;

namespace {

auto distinct_sorted(cudf::column_view const& input,
                     cudf::null_equality nulls_equal = NULL_EQUAL,
                     cudf::nan_equality nans_equal   = NAN_EQUAL)
{
  auto const results = cudf::lists::distinct(lists_cv{input}, nulls_equal, nans_equal);

  // The sorted result will have nulls first and NaNs last.
  // In addition, row equality comparisons in tests just ignore NaN sign thus the expected values
  // can be just NaN while the input can be mixed of NaN and neg_NaN.
  return cudf::lists::sort_lists(
    lists_cv{*results}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
}

}  // namespace

struct ListDistinctTest : public cudf::test::BaseFixture {
};

template <typename T>
struct ListDistinctTypedTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::
  Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes, cudf::test::ChronoTypes>;

TYPED_TEST_SUITE(ListDistinctTypedTest, TestTypes);

TEST_F(ListDistinctTest, TrivialTest)
{
  auto const input =
    floats_lists{{floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 0.0}, null_at(6)},
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)},
                  {} /*NULL*/,
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)}},
                 null_at(2)};

  auto const expected       = floats_lists{{floats_lists{{null, 0.0, 5.0, NaN}, null_at(0)},
                                      floats_lists{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)},
                                      floats_lists{} /*NULL*/,
                                      floats_lists{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)}},
                                     null_at(2)};
  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithSignedZero)
{
  // -0.0 and 0.0 should be considered equal.
  auto const input    = floats_lists{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0, 3};
  auto const expected = floats_lists{0, 1, 2, 3};
  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithInf)
{
  auto const input          = floats_lists{Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf};
  auto const expected       = floats_lists{neg_Inf, 0, Inf};
  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithNaNs)
{
  auto const input =
    floats_lists{0, -1, 1, NaN, 2, 0, neg_NaN, 1, -2, 2, 0, 1, 2, neg_NaN, NaN, NaN, NaN, neg_NaN};

  // NaNs are equal.
  {
    auto const expected       = floats_lists{-2, -1, 0, 1, 2, NaN};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // NaNs are unequal.
  {
    auto const expected       = floats_lists{-2, -1, 0, 1, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN};
    auto const results_sorted = distinct_sorted(input, NULL_EQUAL, NAN_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsNonNull)
{
  // Trivial cases - empty input.
  {
    auto const input          = strings_lists{{}};
    auto const expected       = strings_lists{{}};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // No duplicate.
  {
    auto const input          = strings_lists{"this", "is", "a", "string"};
    auto const expected       = strings_lists{"a", "is", "string", "this"};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // One list column.
  {
    auto const input          = strings_lists{"this", "is", "is", "is", "a", "string", "string"};
    auto const expected       = strings_lists{"a", "is", "string", "this"};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column.
  {
    auto const input = strings_lists{
      strings_lists{"this", "is", "a", "no duplicate", "string"},
      strings_lists{"this", "is", "is", "a", "one duplicate", "string"},
      strings_lists{"this", "is", "is", "is", "a", "two duplicates", "string"},
      strings_lists{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}};
    auto const expected =
      strings_lists{strings_lists{"a", "is", "no duplicate", "string", "this"},
                    strings_lists{"a", "is", "one duplicate", "string", "this"},
                    strings_lists{"a", "is", "string", "this", "two duplicates"},
                    strings_lists{"a", "is", "string", "this", "three duplicates"}};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsWithNullsEqual)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const input = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const expected       = strings_lists{{null, "a", "is", "string", "this"}, null_at(0)};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const input = strings_lists{
      {strings_lists{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                     nulls_at({1, 3, 5, 7})},
       strings_lists{}, /* NULL */
       strings_lists{"this", "is", "is", "a", "one duplicate", "string"}},
      null_at(1)};
    auto const expected =
      strings_lists{{strings_lists{{null, "a", "is", "no duplicate", "string", "this"}, null_at(0)},
                     strings_lists{}, /* NULL */
                     strings_lists{"a", "is", "one duplicate", "string", "this"}},
                    null_at(1)};
    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsWithNullsUnequal)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const input = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const expected =
      strings_lists{{null, null, null, "a", "is", "string", "this"}, nulls_at({0, 1, 2})};
    auto const results_sorted = distinct_sorted(input, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const input = strings_lists{
      {strings_lists{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                     nulls_at({1, 3, 5, 7})},
       strings_lists{}, /* NULL */
       strings_lists{"this", "is", "is", "a", "one duplicate", "string"}},
      null_at(1)};
    auto const expected = strings_lists{
      {strings_lists{{null, null, null, null, "a", "is", "no duplicate", "string", "this"},
                     nulls_at({0, 1, 2, 3})},
       strings_lists{}, /* NULL */
       strings_lists{"a", "is", "one duplicate", "string", "this"}},
      null_at(1)};
    auto const results_sorted = distinct_sorted(input, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}
