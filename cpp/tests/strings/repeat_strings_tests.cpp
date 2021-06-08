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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace {
using STR_COL = cudf::test::strings_column_wrapper;

constexpr bool print_all{false};

auto all_nulls() { return cudf::test::iterator_all_nulls(); }

auto null_at(cudf::size_type idx) { return cudf::test::iterator_with_null_at(idx); }

auto null_at(std::vector<cudf::size_type> const& indices)
{
  return cudf::test::iterator_with_null_at(cudf::host_span<cudf::size_type const>{indices});
}

}  // namespace

struct RepeatJoinStringTest : public cudf::test::BaseFixture {
};

TEST_F(RepeatJoinStringTest, InvalidStringScalar)
{
  auto const str    = cudf::string_scalar("", false);
  auto const result = cudf::strings::repeat_strings(str, 3);
  EXPECT_EQ(result->is_valid(), false);
}

TEST_F(RepeatJoinStringTest, ZeroSizeStringScalar)
{
  auto const str    = cudf::string_scalar("");
  auto const result = cudf::strings::repeat_strings(str, 3);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(result->size(), 0);
}

TEST_F(RepeatJoinStringTest, ValidStringScalar)
{
  auto const str = cudf::string_scalar("abc123xyz-");

  {
    auto const result   = cudf::strings::repeat_strings(str, 3);
    auto const expected = cudf::string_scalar("abc123xyz-abc123xyz-abc123xyz-");
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), result->data(), expected.size());
  }

  // Repeat once.
  {
    auto const result = cudf::strings::repeat_strings(str, 1);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(str.data(), result->data(), str.size());
  }

  // Zero repeat times.
  {
    auto const result = cudf::strings::repeat_strings(str, 0);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Negatitve repeat times.
  {
    auto const result = cudf::strings::repeat_strings(str, -10);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Repeat too many times.
  {
    EXPECT_THROW(cudf::strings::repeat_strings(str, std::numeric_limits<int32_t>::max() / 2),
                 cudf::logic_error);
  }
}

TEST_F(RepeatJoinStringTest, ZeroSizeStringsColumn)
{
  auto const strs    = STR_COL{};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TEST_F(RepeatJoinStringTest, AllEmptyStringsColumn)
{
  auto const strs    = STR_COL{"", "", "", "", ""};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TEST_F(RepeatJoinStringTest, AllNullStringsColumn)
{
  auto const strs    = STR_COL{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TEST_F(RepeatJoinStringTest, ZeroSizeAndNullStringsColumn)
{
  auto const strs =
    STR_COL{{"" /*NULL*/, "", "" /*NULL*/, "", "", "" /*NULL*/}, null_at({0, 2, 5})};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
}

TEST_F(RepeatJoinStringTest, StringsColumnNoNull)
{
  auto const strs = STR_COL{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 2);
    auto const expected = STR_COL{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // Non-positive repeat times.
  {
    auto const expected = STR_COL{"", "", "", "", ""};

    auto results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);

    results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {1, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{"abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 5})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{"xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TEST_F(RepeatJoinStringTest, StringsColumnWithNulls)
{
  auto const strs = STR_COL{{"0a0b0c",
                             "" /*NULL*/,
                             "abcxyz",
                             "" /*NULL*/,
                             "xyzééé",
                             "" /*NULL*/,
                             "ááá",
                             "íí",
                             "",
                             "Hello World"},
                            null_at({1, 3, 5})};

  {
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 2);
    auto const expected = STR_COL{{"0a0b0c0a0b0c",
                                   "" /*NULL*/,
                                   "abcxyzabcxyz",
                                   "" /*NULL*/,
                                   "xyzéééxyzééé",
                                   "" /*NULL*/,
                                   "áááááá",
                                   "íííí",
                                   "",
                                   "Hello WorldHello World"},
                                  null_at({1, 3, 5})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // Non-positive repeat times.
  {
    auto const expected = STR_COL{
      {"", "" /*NULL*/, "", "" /*NULL*/, "", "" /*NULL*/, "", "", "", ""}, null_at({1, 3, 5})};

    auto results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);

    results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{{"0a0b0c0a0b0c", "" /*NULL*/, "abcxyzabcxyz"}, null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 7})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{
      {"abcxyzabcxyz", "" /*NULL*/, "xyzéééxyzééé", "" /*NULL*/, "áááááá"}, null_at({1, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {6, 10})[0];
    auto const results  = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected = STR_COL{"áááááá", "íííí", "", "Hello WorldHello World"};

    // The results strings column may have a bitmask with all valid values.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, print_all);
  }
}
