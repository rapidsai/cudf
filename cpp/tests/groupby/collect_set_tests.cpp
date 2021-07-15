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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {

#define COL_K    cudf::test::fixed_width_column_wrapper<int32_t, int32_t>
#define COL_V    cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>
#define COL_S    cudf::test::strings_column_wrapper
#define LCL_V    cudf::test::lists_column_wrapper<TypeParam, int32_t>
#define LCL_S    cudf::test::lists_column_wrapper<cudf::string_view>
#define VALIDITY std::initializer_list<bool>

struct CollectSetTest : public cudf::test::BaseFixture {
  static auto collect_set() { return cudf::make_collect_set_aggregation(); }

  static auto collect_set_null_unequal()
  {
    return cudf::make_collect_set_aggregation(null_policy::INCLUDE, null_equality::UNEQUAL);
  }

  static auto collect_set_null_exclude()
  {
    return cudf::make_collect_set_aggregation(null_policy::EXCLUDE);
  }
};

template <typename V>
struct CollectSetTypedTest : public cudf::test::BaseFixture {
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(CollectSetTypedTest, FixedWidthTypesNotBool);

TYPED_TEST(CollectSetTypedTest, TrivialInput)
{
  // Empty input
  test_single_agg(COL_K{}, COL_V{}, COL_K{}, LCL_V{}, CollectSetTest::collect_set());

  // Single key input
  {
    COL_K keys{1};
    COL_V vals{10};
    COL_K keys_expected{1};
    LCL_V vals_expected{LCL_V{10}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
  }

  // Non-repeated keys
  {
    COL_K keys{2, 1};
    COL_V vals{20, 10};
    COL_K keys_expected{1, 2};
    LCL_V vals_expected{LCL_V{10}, LCL_V{20}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
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
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    COL_K keys{4, 1, 2, 4, 3, 3, 2, 1};
    COL_V vals{40, 10, 20, 40, 30, 30, 20, 11};
    COL_K keys_expected{1, 2, 3, 4};
    LCL_V vals_expected{{10, 11}, {20}, {30}, {40}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
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
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
  }
  {
    auto const keys = cudf::slice(keys_original, {2, 10})[0];  // { 1, 1, 2, 2, 2, 2, 3, 3 }
    auto const vals = cudf::slice(vals_original, {2, 10})[0];  // { 10, 10, 20, 21, 21, 20, 30, 33 }
    auto const keys_expected = COL_K{1, 2, 3};
    auto const vals_expected = LCL_V{{10}, {20, 21}, {30, 33}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
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
  test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
}

TEST_F(CollectSetTest, FloatsWithNaN)
{
  COL_K keys{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<float> vals{
    {1.0f, 1.0f, -2.3e-5f, -2.3e-5f, 2.3e5f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
    {true, true, true, true, true, true, true, true, true, true, false, false}};
  COL_K keys_expected{1};
  // null equal with nan unequal
  cudf::test::lists_column_wrapper<float> vals_expected{
    {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f},
     VALIDITY{true, true, true, true, true, true, true, false}},
  };
  test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());
  // null unequal with nan unequal
  vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
                    VALIDITY{true, true, true, true, true, true, true, false, false}}};
  test_single_agg(
    keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_unequal());
  // null exclude with nan unequal
  vals_expected = {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN}};
  test_single_agg(
    keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_exclude());
  // null equal with nan equal
  vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, NAN, 0.0f}, VALIDITY{true, true, true, true, false}}};
  test_single_agg(keys,
                  vals,
                  keys_expected,
                  vals_expected,
                  cudf::make_collect_set_aggregation(
                    null_policy::INCLUDE, null_equality::EQUAL, nan_equality::ALL_EQUAL));
  // null unequal with nan equal
  vals_expected = {
    {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f, 0.0f}, VALIDITY{true, true, true, true, false, false}}};
  test_single_agg(keys,
                  vals,
                  keys_expected,
                  vals_expected,
                  cudf::make_collect_set_aggregation(
                    null_policy::INCLUDE, null_equality::UNEQUAL, nan_equality::ALL_EQUAL));
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

    // By default, nulls are consider equals, thus only one null is kept per key
    LCL_V vals_expected{{{10, null}, VALIDITY{true, false}},
                        {{20, null}, VALIDITY{true, false}},
                        {{30, 31}, VALIDITY{true, true}}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());

    // All nulls per key are kept (nulls are put at the end of each list)
    vals_expected = LCL_V{{{10, null, null}, VALIDITY{true, false, false}},
                          {{20, null, null, null}, VALIDITY{true, false, false, false}},
                          {{30, 31}, VALIDITY{true, true}}};
    test_single_agg(
      keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_unequal());

    // All nulls per key are excluded
    vals_expected = LCL_V{{10}, {20}, {30, 31}};
    test_single_agg(
      keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_exclude());
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    COL_K keys{4, 1, 2, 4, 3, 3, 3, 3, 2, 1};
    COL_V vals{{40, 10, 20, 40, null, null, null, null, 21, null},
               {true, true, true, true, false, false, false, false, true, false}};
    COL_K keys_expected{1, 2, 3, 4};

    // By default, nulls are consider equals, thus only one null is kept per key
    LCL_V vals_expected{{{10, null}, VALIDITY{true, false}},
                        {{20, 21}, VALIDITY{true, true}},
                        {{null}, VALIDITY{false}},
                        {{40}, VALIDITY{true}}};
    test_single_agg(keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set());

    // All nulls per key are kept (nulls are put at the end of each list)
    vals_expected = LCL_V{{{10, null}, VALIDITY{true, false}},
                          {{20, 21}, VALIDITY{true, true}},
                          {{null, null, null, null}, VALIDITY{false, false, false, false}},
                          {{40}, VALIDITY{true}}};
    test_single_agg(
      keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_unequal());

    // All nulls per key are excluded
    vals_expected = LCL_V{{10}, {20, 21}, {}, {40}};
    test_single_agg(
      keys, vals, keys_expected, vals_expected, CollectSetTest::collect_set_null_exclude());
  }
}

}  // namespace test
}  // namespace cudf
