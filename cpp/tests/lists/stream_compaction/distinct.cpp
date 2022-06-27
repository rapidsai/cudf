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

auto constexpr null{0};  // null at current level
// auto constexpr XXX{0};   // null pushed down from parent level
auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();

using bools_col = cudf::test::fixed_width_column_wrapper<bool>;
// using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
// using floats_col  = cudf::test::fixed_width_column_wrapper<float>;
using lists_col = cudf::test::lists_column_wrapper<float>;
// using strings_col = cudf::test::strings_column_wrapper;
// using structs_col = cudf::test::structs_column_wrapper;
using lists_cv = cudf::lists_column_view;

// using cudf::nan_policy;
// using cudf::null_equality;
// using cudf::null_policy;
// using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
// using cudf::test::iterators::nulls_at;

namespace {
auto distinct_sorted(cudf::column_view const& input)
{
  auto const results = cudf::lists::distinct(lists_cv{input});
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
  auto const input = lists_col{{lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 0.0}, null_at(6)},
                                lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)},
                                {} /*NULL*/,
                                lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)}},
                               null_at(2)};

  auto const results_sorted = distinct_sorted(input);
  auto const expected       = lists_col{{lists_col{{null, 0.0, 5.0, NaN}, null_at(0)},
                                   lists_col{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)},
                                   lists_col{} /*NULL*/,
                                   lists_col{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)}},
                                  null_at(2)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}
