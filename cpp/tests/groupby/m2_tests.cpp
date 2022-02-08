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

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};                                       // Mark for null elements
constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};  // Mark for NaN double elements

template <class T>
using keys_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

template <class T>
using vals_col = cudf::test::fixed_width_column_wrapper<T>;

using double_col  = vals_col<double>;
using structs_col = cudf::test::structs_column_wrapper;

auto compute_M2(cudf::column_view const& keys, cudf::column_view const& values)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = values;
  requests[0].aggregations.emplace_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));
  auto result = gb_obj.aggregate(requests);
  return std::make_pair(std::move(result.first->release()[0]),
                        std::move(result.second[0].results[0]));
}
}  // namespace

template <class T>
struct GroupbyM2TypedTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                     cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(GroupbyM2TypedTest, TestTypes);

TYPED_TEST(GroupbyM2TypedTest, EmptyInput)
{
  using T = TypeParam;

  auto const keys = keys_col<T>{};
  auto const vals = vals_col<T>{};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_vals        = structs_col{};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, AllNullKeysInput)
{
  using T = TypeParam;

  auto const keys = keys_col<T>{{1, 2, 3}, all_nulls()};
  auto const vals = vals_col<T>{3, 4, 5};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{};
  auto const expected_vals        = structs_col{};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, AllNullValuesInput)
{
  using T = TypeParam;

  auto const keys = keys_col<T>{1, 2, 3};
  auto const vals = vals_col<T>{{3, 4, 5}, all_nulls()};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_vals        = [] {
    auto counts = double_col{0, 0, 0};
    auto means  = double_col{{null, null, null}, all_nulls()};
    auto m2s    = double_col{{null, null, null}, all_nulls()};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SimpleInput)
{
  using T = TypeParam;

  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, 5, 9]
  // key = 3: vals = [2, 7, 8]
  auto const keys = keys_col<T>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = vals_col<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3};
  auto const expected_vals        = [] {
    auto counts = double_col{3, 4, 3};
    auto means  = double_col{3.0, 4.75, 17.0 / 3.0};
    auto m2s    = double_col{18.0, 32.75, 20.0 + 2.0 / 3.0};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SimpleInputHavingNegativeValues)
{
  using T = TypeParam;

  // key = 1: vals = [0,  3, -6]
  // key = 2: vals = [1, -4, -5, 9]
  // key = 3: vals = [-2, 7, -8]
  auto const keys = keys_col<T>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = vals_col<T>{0, 1, -2, 3, -4, -5, -6, 7, -8, 9};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3};
  auto const expected_vals        = [] {
    auto counts = double_col{3, 4, 3};
    auto means  = double_col{-1.0, 0.25, -1.0};
    auto m2s    = double_col{42.0, 122.75, 114.0};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, ValuesHaveNulls)
{
  using T = TypeParam;

  // key = 1: vals = [0]
  // key = 2: vals = [null, 5, 7]
  // key = 3: vals = [2, 6]
  // key = 4: vals = [3]
  // key = 5: vals = [null]
  auto const keys = keys_col<T>{1, 2, 3, 4, 5, 2, 3, 2};
  auto const vals = vals_col<T>{{0, null, 2, 3, null, 5, 6, 7}, nulls_at({1, 4})};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3, 4, 5};
  auto const expected_vals        = [] {
    auto counts = double_col{1, 2, 2, 1, 0};
    auto means  = double_col{{0, 6, 4, 3, null}, null_at(4)};
    auto m2s    = double_col{{0.0, 2.0, 8.0, 0.0, 0.0 /*NULL*/}, null_at(4)};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, KeysAndValuesHaveNulls)
{
  using T = TypeParam;

  // key = 1: vals = [null, 3, 6]
  // key = 2: vals = [1, 4, null, 9]
  // key = 3: vals = [2, 8]
  // key = 4: vals = [null]
  auto const keys = keys_col<T>{{1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = vals_col<T>{{null, 1, 2, 3, 4, null, 6, 7, 8, 9, null}, nulls_at({0, 5, 10})};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3, 4};
  auto const expected_vals        = [] {
    auto counts = double_col{2, 3, 2, 0};
    auto means  = double_col{{4.5, 14.0 / 3.0, 5.0, 0.0 /*NULL*/}, null_at(3)};
    auto m2s    = double_col{{4.5, 32.0 + 2.0 / 3.0, 18.0, 0.0 /*NULL*/}, null_at(3)};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, InputHaveNullsAndNaNs)
{
  using T = TypeParam;

  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, NaN, 9]
  // key = 3: vals = [null, 2, 8]
  // key = 4: vals = [null, 10, NaN]
  auto const keys = keys_col<T>{{4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4}, null_at(9)};
  auto const vals = vals_col<double>{
    {0.0 /*NULL*/, 0.0 /*NULL*/, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, 9.0, 10.0, NaN},
    nulls_at({0, 1})};

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3, 4};
  auto const expected_vals        = [] {
    auto counts = double_col{3, 4, 2, 2};
    auto means  = double_col{3.0, NaN, 5.0, NaN};
    auto m2s    = double_col{18.0, NaN, 18.0, NaN};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SlicedColumnsInput)
{
  using T = TypeParam;

  // This test should compute M2 aggregation on the same dataset as the InputHaveNullsAndNaNs test.
  // i.e.:
  //
  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, NaN, 9]
  // key = 3: vals = [null, 2, 8]
  // key = 4: vals = [null, 10, NaN]

  auto const keys_original =
    keys_col<T>{{
                  1, 2, 3, 4, 5, 1, 2, 3, 4, 5,                 // will not use, don't care
                  4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4,  // use this
                  1, 2, 3, 4, 5, 1, 2, 3, 4, 5                  // will not use, don't care
                },
                null_at(19)};
  auto const vals_original = vals_col<double>{
    {
      3.0, 2.0,  5.0,  4.0,  6.0, 9.0, 1.0, 0.0,  1.0,  7.0,  // will not use, don't care
      0.0, 0.0,  0.0,  1.0,  2.0, 3.0, 4.0, NaN,  6.0,  7.0, 8.0, 9.0, 10.0, NaN,  // use this
      9.0, 10.0, 11.0, 12.0, 0.0, 5.0, 1.0, 20.0, 19.0, 15.0  // will not use, don't care
    },
    nulls_at({10, 11})};

  auto const keys = cudf::slice(keys_original, {10, 24})[0];
  auto const vals = cudf::slice(vals_original, {10, 24})[0];

  auto const [out_keys, out_vals] = compute_M2(keys, vals);
  auto const expected_keys        = keys_col<T>{1, 2, 3, 4};
  auto const expected_vals        = [] {
    auto counts = double_col{3, 4, 2, 2};
    auto means  = double_col{3.0, NaN, 5.0, NaN};
    auto m2s    = double_col{18.0, NaN, 18.0, NaN};
    return structs_col{{counts, means, m2s}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_vals, *out_vals, verbosity);
}
