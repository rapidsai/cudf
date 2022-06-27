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
auto constexpr neg_NaN     = -std::numeric_limits<float_type>::quiet_NaN();
auto constexpr neg_Inf     = -std::numeric_limits<float_type>::infinity();
auto constexpr NaN         = std::numeric_limits<float_type>::quiet_NaN();
auto constexpr Inf         = std::numeric_limits<float_type>::infinity();
auto constexpr NAN_EQUAL   = cudf::nan_equality::ALL_EQUAL;
auto constexpr NAN_UNEQUAL = cudf::nan_equality::UNEQUAL;

using bools_col = cudf::test::fixed_width_column_wrapper<bool>;
// using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col    = cudf::test::fixed_width_column_wrapper<float_type>;
using floats_lists  = cudf::test::lists_column_wrapper<float_type>;
using strings_lists = cudf::test::lists_column_wrapper<cudf::string_view>;
// using strings_col = cudf::test::strings_column_wrapper;
// using structs_col = cudf::test::structs_column_wrapper;
using lists_cv = cudf::lists_column_view;

namespace {

auto distinct_sorted(cudf::column_view const& input, cudf::nan_equality nans_equal = NAN_EQUAL)
{
  auto const results =
    cudf::lists::distinct(lists_cv{input}, cudf::null_equality::EQUAL, nans_equal);

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
  auto const input  = floats_lists{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0, 3};
  auto const expect = floats_lists{0, 1, 2, 3};
  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *results_sorted);
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
    auto const results_sorted = distinct_sorted(input, NAN_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}
