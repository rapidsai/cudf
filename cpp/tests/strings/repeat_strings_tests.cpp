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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>

using namespace cudf::test::iterators;

namespace {
using StrCol = cudf::test::strings_column_wrapper;

constexpr int32_t null{0};  // mark for null elements in a column of int32_t values
constexpr bool print_all{false};
}  // namespace

template <typename T>
struct RepeatStringsTypedTest : public cudf::test::BaseFixture {
};

// Test for signed types only, as we will need to use non-positive values.
using TypesForTest = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(RepeatStringsTypedTest, TypesForTest);

TYPED_TEST(RepeatStringsTypedTest, InvalidStringScalar)
{
  auto const str    = cudf::string_scalar("", false);
  auto const result = cudf::strings::repeat_string(str, 3);
  EXPECT_EQ(result->is_valid(), false);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringScalar)
{
  auto const str    = cudf::string_scalar("");
  auto const result = cudf::strings::repeat_string(str, 3);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(result->size(), 0);
}

TYPED_TEST(RepeatStringsTypedTest, ValidStringScalar)
{
  auto const str = cudf::string_scalar("abc123xyz-");

  {
    auto const result   = cudf::strings::repeat_string(str, 3);
    auto const expected = cudf::string_scalar("abc123xyz-abc123xyz-abc123xyz-");
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), result->data(), expected.size());
  }

  // Repeat once.
  {
    auto const result = cudf::strings::repeat_string(str, 1);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(str.data(), result->data(), str.size());
  }

  // Zero repeat times.
  {
    auto const result = cudf::strings::repeat_string(str, 0);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Negatitve repeat times.
  {
    auto const result = cudf::strings::repeat_string(str, -10);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Repeat too many times.
  {
    EXPECT_THROW(cudf::strings::repeat_string(str, std::numeric_limits<int32_t>::max() / 2),
                 cudf::logic_error);
  }
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = StrCol{};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringsColumnWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = StrCol{};
  auto const repeat_times = IntCol{};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, AllEmptyStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = StrCol{"", "", "", "", ""};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, AllEmptyStringsColumnWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = StrCol{"", "", "", "", ""};
  auto const repeat_times = IntCol{-2, -1, 0, 1, 2};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, AllNullStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = StrCol{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, AllNullStringsColumnWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = StrCol{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};

  // The repeat_times column contains all valid numbers.
  {
    auto const repeat_times = IntCol{-1, 0, 1};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // The repeat_times column also contains some nulls and some valid numbers.
  {
    auto const repeat_times = IntCol{{null, 1, null}, nulls_at({0, 2})};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // The repeat_times column also contains all nulls.
  {
    auto const repeat_times = IntCol{{null, null, null}, all_nulls()};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithAllNullColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = StrCol{"ABC", "abc", "xyz"};
  auto const repeat_times = IntCol{{null, null, null}, all_nulls()};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  auto const expected = StrCol{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeAndNullStringsColumnWithScalarRepeatTimes)
{
  auto const strs =
    StrCol{{"" /*NULL*/, "", "" /*NULL*/, "", "", "" /*NULL*/}, nulls_at({0, 2, 5})};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeAndNullStringsColumnWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs =
    StrCol{{"" /*NULL*/, "", "" /*NULL*/, "", "", "" /*NULL*/}, nulls_at({0, 2, 5})};
  auto const repeat_times = IntCol{1, 2, 3, 4, 5, 6};
  auto const results      = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithInvalidColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = StrCol{"abc", "xyz"};

  // Size mismatched.
  {
    auto const repeat_times = IntCol{1, 2, 3, 4, 5, 6};
    EXPECT_THROW(cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times),
                 cudf::logic_error);
  }

  // Invalid data type.
  {
    auto const repeat_times = cudf::test::fixed_width_column_wrapper<float>{1, 2, 3, 4, 5, 6};
    EXPECT_THROW(cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times),
                 cudf::logic_error);
  }

  // Invalid data type.
  {
    auto const repeat_times = StrCol{"xxx", "xxx"};
    EXPECT_THROW(cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times),
                 cudf::logic_error);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnNoNullWithScalarRepeatTimes)
{
  auto const strs = StrCol{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 2);
    auto const expected = StrCol{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // Non-positive repeat times.
  {
    auto const expected = StrCol{"", "", "", "", ""};

    auto results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);

    results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnNoNullWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = StrCol{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  // Repeat once.
  {
    auto const repeat_times = IntCol{1, 1, 1, 1, 1};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // repeat_times column has negative values.
  {
    auto const repeat_times = IntCol{1, 2, 3, -1, -2};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    auto const expected = StrCol{"0a0b0c", "abcxyzabcxyz", "xyzéééxyzéééxyzééé", "", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // repeat_times column has nulls.
  {
    auto const repeat_times = IntCol{{1, null, 3, 2, null}, nulls_at({1, 4})};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    auto const expected = StrCol{
      {"0a0b0c", "" /*NULL*/, "xyzéééxyzéééxyzééé", "áááááá", "" /*NULL*/}, nulls_at({1, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnNoNullWithScalarRepeatTimes)
{
  auto const strs = StrCol{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {1, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{"abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 5})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{"xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnNoNullWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = StrCol{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};
  auto const repeat_times = IntCol{1, 2, 3, 2, 3};

  // Sliced the first half of the column.
  {
    auto const sliced_strs   = cudf::slice(strs, {0, 3})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {0, 3})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{"0a0b0c", "abcxyzabcxyz", "xyzéééxyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs   = cudf::slice(strs, {1, 3})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {1, 3})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{"abcxyzabcxyz", "xyzéééxyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs   = cudf::slice(strs, {2, 5})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {2, 5})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{"xyzéééxyzéééxyzééé", "áááááá", "íííííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithNullsWithScalarRepeatTimes)
{
  auto const strs = StrCol{{"0a0b0c",
                            "" /*NULL*/,
                            "abcxyz",
                            "" /*NULL*/,
                            "xyzééé",
                            "" /*NULL*/,
                            "ááá",
                            "íí",
                            "",
                            "Hello World"},
                           nulls_at({1, 3, 5})};

  {
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 2);
    auto const expected = StrCol{{"0a0b0c0a0b0c",
                                  "" /*NULL*/,
                                  "abcxyzabcxyz",
                                  "" /*NULL*/,
                                  "xyzéééxyzééé",
                                  "" /*NULL*/,
                                  "áááááá",
                                  "íííí",
                                  "",
                                  "Hello WorldHello World"},
                                 nulls_at({1, 3, 5})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // Non-positive repeat times.
  {
    auto const expected = StrCol{
      {"", "" /*NULL*/, "", "" /*NULL*/, "", "" /*NULL*/, "", "", "", ""}, nulls_at({1, 3, 5})};

    auto results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);

    results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithNullsWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = StrCol{{"0a0b0c",
                            "" /*NULL*/,
                            "abcxyz",
                            "" /*NULL*/,
                            "xyzééé",
                            "" /*NULL*/,
                            "ááá",
                            "íí",
                            "",
                            "Hello World"},
                           nulls_at({1, 3, 5})};

  // Repeat once.
  {
    auto const repeat_times = IntCol{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // repeat_times column has negative values.
  {
    auto const repeat_times = IntCol{1, 2, 3, -1, -2, 1, 2, 3, -5, 0};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    auto const expected = StrCol{{"0a0b0c",
                                  "" /*NULL*/,
                                  "abcxyzabcxyzabcxyz",
                                  "" /*NULL*/,
                                  "",
                                  "" /*NULL*/,
                                  "áááááá",
                                  "íííííí",
                                  "",
                                  ""},
                                 nulls_at({1, 3, 5})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // repeat_times column has nulls.
  {
    auto const repeat_times =
      IntCol{{1, 2, null, -1, null, 1, 2, null, -5, 0}, nulls_at({2, 4, 7})};
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
    auto const expected = StrCol{{"0a0b0c",
                                  "" /*NULL*/,
                                  "" /*NULL*/,
                                  "" /*NULL*/,
                                  "" /*NULL*/,
                                  "" /*NULL*/,
                                  "áááááá",
                                  "" /*NULL*/,
                                  "",
                                  ""},
                                 nulls_at({1, 2, 3, 4, 5, 7})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnWithNullsWithScalarRepeatTimes)
{
  auto const strs = StrCol{{"0a0b0c",
                            "" /*NULL*/,
                            "abcxyz",
                            "" /*NULL*/,
                            "xyzééé",
                            "" /*NULL*/,
                            "ááá",
                            "íí",
                            "",
                            "Hello World"},
                           nulls_at({1, 3, 5})};

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{{"0a0b0c0a0b0c", "" /*NULL*/, "abcxyzabcxyz"}, null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 7})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{
      {"abcxyzabcxyz", "" /*NULL*/, "xyzéééxyzééé", "" /*NULL*/, "áááááá"}, nulls_at({1, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {6, 10})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = StrCol{"áááááá", "íííí", "", "Hello WorldHello World"};

    // The results strings column may have a bitmask with all valid values.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, print_all);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnWithNullsWithColumnRepeatTimes)
{
  using IntCol = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = StrCol{{"0a0b0c",
                            "" /*NULL*/,
                            "abcxyz",
                            "" /*NULL*/,
                            "xyzééé",
                            "" /*NULL*/,
                            "ááá",
                            "íí",
                            "",
                            "Hello World"},
                           nulls_at({1, 3, 5})};

  auto const repeat_times = IntCol{{1, 2, null, -1, null, 1, 2, null, -5, 0}, nulls_at({2, 4, 7})};

  // Sliced the first half of the column.
  {
    auto const sliced_strs   = cudf::slice(strs, {0, 3})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {0, 3})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{{"0a0b0c", "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs   = cudf::slice(strs, {2, 7})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {2, 7})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/, "" /*NULL*/, "áááááá"},
                                 nulls_at({0, 1, 2, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column, output has nulls.
  {
    auto const sliced_strs   = cudf::slice(strs, {6, 10})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {6, 10})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{{"áááááá", "" /*NULL*/, "", ""}, null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column, output does not have null.
  {
    auto const sliced_strs   = cudf::slice(strs, {8, 10})[0];
    auto const sliced_rtimes = cudf::slice(repeat_times, {8, 10})[0];
    auto const results =
      cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), sliced_rtimes);
    auto const expected = StrCol{"", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}
