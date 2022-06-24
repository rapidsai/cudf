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

#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/sorting.hpp>

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
using cudf::test::iterators::nulls_at;

namespace {
auto set_op_sorted(cudf::column_view const& lhs, cudf::column_view const& rhs)
{
  auto const results = cudf::lists::set_difference(lists_cv{lhs}, lists_cv{rhs});
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
  auto const lhs = lists_col{{lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 0.0}, null_at(6)},
                              lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)},
                              {} /*NULL*/,
                              lists_col{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)}},
                             null_at(2)};
  auto const rhs = lists_col{{lists_col{{1.0, 0.5, null, 0.0, 0.0, null, NaN}, nulls_at({2, 5})},
                              lists_col{{2.0, 1.0, null, 0.0, 0.0, null}, nulls_at({2, 5})},
                              lists_col{{2.0, 1.0, null, 0.0, 0.0, null}, nulls_at({2, 5})},
                              {} /*NULL*/},
                             null_at(3)};

  auto const results_sorted = set_op_sorted(lhs, rhs);
  auto const expected =
    lists_col{{lists_col{5.0}, lists_col{5.0, NaN}, lists_col{} /*NULL*/, lists_col{} /*NULL*/},
              nulls_at({2, 3})};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results_sorted);
}
