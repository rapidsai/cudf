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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace test {

namespace {

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

#define COL_K    cudf::test::fixed_width_column_wrapper<int32_t, int32_t>
#define COL_V    cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>
#define COL_S    cudf::test::strings_column_wrapper
#define LCL_V    cudf::test::lists_column_wrapper<TypeParam, int32_t>
#define LCL_S    cudf::test::lists_column_wrapper<cudf::string_view>
#define VALIDITY std::initializer_list<bool>

auto groupby_collect_set(cudf::column_view const& keys,
                         cudf::column_view const& values,
                         std::unique_ptr<groupby_aggregation>&& agg)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = values;
  requests[0].aggregations.emplace_back(std::move(agg));

  auto const result     = cudf::groupby::groupby(cudf::table_view({keys})).aggregate(requests);
  auto const sort_order = cudf::sorted_order(result.first->view(), {}, {cudf::null_order::AFTER});
  auto const sorted_vals =
    std::move(cudf::gather(cudf::table_view{{result.second[0].results[0]->view()}}, *sort_order)
                ->release()
                .front());

  auto result_keys = std::move(cudf::gather(result.first->view(), *sort_order)->release().front());
  auto result_vals = cudf::lists::sort_lists(
    cudf::lists_column_view{sorted_vals->view()}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  return std::pair(std::move(result_keys), std::move(result_vals));
}

}  // namespace

struct CollectSetTest : public cudf::test::BaseFixture {
  static auto collect_set()
  {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>();
  }

  static auto collect_set_null_unequal()
  {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(null_policy::INCLUDE,
                                                                         null_equality::UNEQUAL);
  }

  static auto collect_set_null_exclude()
  {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(null_policy::EXCLUDE);
  }
};

template <typename V>
struct CollectSetTypedTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(CollectSetTypedTest, FixedWidthTypesNotBool);

TYPED_TEST(CollectSetTypedTest, TrivialInput)
{
  // Empty input
  {
    COL_K keys{};
    COL_V vals{};
    COL_K keys_expected{};
    LCL_V vals_expected{};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Single key input
  {
    COL_K keys{1};
    COL_V vals{10};
    COL_K keys_expected{1};
    LCL_V vals_expected{LCL_V{10}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Non-repeated keys
  {
    COL_K keys{2, 1};
    COL_V vals{20, 10};
    COL_K keys_expected{1, 2};
    LCL_V vals_expected{LCL_V{10}, LCL_V{20}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TYPED_TEST(CollectSetTypedTest, TypicalInput)
{
  // Pre-sorted keys
  {
    COL_K keys{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    COL_V vals{10, 11, 10, 10, 20, 21, 21, 20, 30, 33, 32, 31};
    COL_K keys_expected{1, 2, 3};
    LCL_V vals_expected{{10, 11}, {20, 21}, {30, 31, 32, 33}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    COL_K keys{4, 1, 2, 4, 3, 3, 2, 1};
    COL_V vals{40, 10, 20, 40, 30, 30, 20, 11};
    COL_K keys_expected{1, 2, 3, 4};
    LCL_V vals_expected{{10, 11}, {20}, {30}, {40}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

// Keys and values columns are sliced columns
TYPED_TEST(CollectSetTypedTest, SlicedColumnsInput)
{
  COL_K keys_original{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  COL_V vals_original{10, 11, 10, 10, 20, 21, 21, 20, 30, 33, 32, 31};
  {
    auto const keys          = cudf::slice(keys_original, {0, 4})[0];  // { 1, 1, 1, 1 }
    auto const vals          = cudf::slice(vals_original, {0, 4})[0];  // { 10, 11, 10, 10 }
    auto const keys_expected = COL_K{1};
    auto const vals_expected = LCL_V{{10, 11}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
  {
    auto const keys = cudf::slice(keys_original, {2, 10})[0];  // { 1, 1, 2, 2, 2, 2, 3, 3 }
    auto const vals = cudf::slice(vals_original, {2, 10})[0];  // { 10, 10, 20, 21, 21, 20, 30, 33 }
    auto const keys_expected = COL_K{1, 2, 3};
    auto const vals_expected = LCL_V{{10}, {20, 21}, {30, 33}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TEST_F(CollectSetTest, StringInput)
{
  COL_K keys{1, 2, 3, 3, 2, 1, 2, 1, 2, 1, 1, 1, 1};
  COL_S vals{
    "String 1, first",
    "String 2, first",
    "String 3, first",
    "String 3, second",
    "String 2, second",
    "String 1, second",
    "String 2, second",  // repeated
    "String 1, second",  // repeated
    "String 2, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second"   // repeated
  };
  COL_K keys_expected{1, 2, 3};
  LCL_S vals_expected{{"String 1, first", "String 1, second"},
                      {"String 2, first", "String 2, second"},
                      {"String 3, first", "String 3, second"}};

  auto const [out_keys, out_lists] = groupby_collect_set(keys, vals, CollectSetTest::collect_set());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
}

TEST_F(CollectSetTest, FloatsWithNaN)
{
  COL_K keys{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<float> vals{
    {1.0f, 1.0f, -2.3e-5f, -2.3e-5f, 2.3e5f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
    {true, true, true, true, true, true, true, true, true, true, false, false}};
  COL_K keys_expected{1};
  cudf::test::lists_column_wrapper<float> vals_expected;

  // null equal with nan unequal
  {
    vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f},
                      VALIDITY{true, true, true, true, true, true, true, false}}};
    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null unequal with nan unequal
  {
    vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
                      VALIDITY{true, true, true, true, true, true, true, false, false}}};
    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_unequal());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null exclude with nan unequal
  {
    vals_expected = {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN}};
    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_exclude());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null equal with nan equal
  {
    vals_expected = {
      {{-2.3e-5f, 1.0f, 2.3e5f, NAN, 0.0f}, VALIDITY{true, true, true, true, false}}};
    auto const [out_keys, out_lists] =
      groupby_collect_set(keys,
                          vals,
                          cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
                            null_policy::INCLUDE, null_equality::EQUAL, nan_equality::ALL_EQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null unequal with nan equal
  {
    vals_expected = {
      {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f, 0.0f}, VALIDITY{true, true, true, true, false, false}}};
    auto const [out_keys, out_lists] =
      groupby_collect_set(keys,
                          vals,
                          cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
                            null_policy::INCLUDE, null_equality::UNEQUAL, nan_equality::ALL_EQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TYPED_TEST(CollectSetTypedTest, CollectWithNulls)
{
  // Just use an arbitrary value to store null entries
  // Using this alias variable will make the code look cleaner
  constexpr int32_t null = 0;

  // Pre-sorted keys
  {
    COL_K keys{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    COL_V vals{{10, 10, null, null, 20, null, null, null, 30, 31, 30, 31},
               {true, true, false, false, true, false, false, false, true, true, true, true}};
    COL_K keys_expected{1, 2, 3};
    LCL_V vals_expected;

    // By default, nulls are consider equals, thus only one null is kept per key
    {
      vals_expected = {{{10, null}, VALIDITY{true, false}},
                       {{20, null}, VALIDITY{true, false}},
                       {{30, 31}, VALIDITY{true, true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are kept (nulls are put at the end of each list)
    {
      vals_expected = LCL_V{{{10, null, null}, VALIDITY{true, false, false}},
                            {{20, null, null, null}, VALIDITY{true, false, false, false}},
                            {{30, 31}, VALIDITY{true, true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_unequal());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are excluded
    {
      vals_expected = LCL_V{{10}, {20}, {30, 31}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_exclude());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    COL_K keys{4, 1, 2, 4, 3, 3, 3, 3, 2, 1};
    COL_V vals{{40, 10, 20, 40, null, null, null, null, 21, null},
               {true, true, true, true, false, false, false, false, true, false}};
    COL_K keys_expected{1, 2, 3, 4};
    LCL_V vals_expected;

    // By default, nulls are consider equals, thus only one null is kept per key
    {
      vals_expected = {{{10, null}, VALIDITY{true, false}},
                       {{20, 21}, VALIDITY{true, true}},
                       {{null}, VALIDITY{false}},
                       {{40}, VALIDITY{true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are kept (nulls are put at the end of each list)
    {
      vals_expected = LCL_V{{{10, null}, VALIDITY{true, false}},
                            {{20, 21}, VALIDITY{true, true}},
                            {{null, null, null, null}, VALIDITY{false, false, false, false}},
                            {{40}, VALIDITY{true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_unequal());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are excluded
    {
      vals_expected = LCL_V{{10}, {20, 21}, {}, {40}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_exclude());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }
  }
}

}  // namespace test
}  // namespace cudf
