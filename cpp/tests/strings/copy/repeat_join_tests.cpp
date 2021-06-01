/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/copy.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

namespace {
using STR_COL = cudf::test::strings_column_wrapper;

constexpr bool print_all{false};

// auto all_nulls() { return cudf::test::iterator_all_nulls(); }

// auto null_at(cudf::size_type idx) { return cudf::test::iterator_with_null_at(idx); }

// auto null_at(std::vector<cudf::size_type> const& indices)
//{
//  return cudf::test::iterator_with_null_at(cudf::host_span<cudf::size_type const>{indices});
//}

// auto nulls_from_nullptr(std::vector<const char*> const& strs)
//{
//  return thrust::make_transform_iterator(strs.begin(), [](auto ptr) { return ptr != nullptr; });
//}

}  // namespace

struct RepeatJoinStringTest : public cudf::test::BaseFixture {
};

TEST_F(RepeatJoinStringTest, InvalidRepeatTimes) {}

TEST_F(RepeatJoinStringTest, InvalidStringScalar) {}

TEST_F(RepeatJoinStringTest, ZeroSizeStringScalar) {}

TEST_F(RepeatJoinStringTest, ValidStringScalar)
{
  auto const str = cudf::string_scalar("abc123xyz-");

  {
    auto const result   = cudf::strings::repeat_join(str, 3);
    auto const expected = cudf::string_scalar("abc123xyz-abc123xyz-abc123xyz-");
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), result.data(), expected.size());
  }

  // Repeat once.
  {
    auto const result = cudf::strings::repeat_join(str, 1);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(str.data(), result.data(), str.size());
  }

  // Zero repeat time.
  {
  }
}

TEST_F(RepeatJoinStringTest, ZeroSizeStringsColumn) {}

TEST_F(RepeatJoinStringTest, AllNullStringsColumn) {}

TEST_F(RepeatJoinStringTest, ZeroSizeAndNullStringsColumn) {}

TEST_F(RepeatJoinStringTest, StringsColumnNoNull)
{
  auto const strs = STR_COL{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  {
    auto const results = cudf::strings::repeat_join(cudf::strings_column_view(strs), 2);
    auto const expected = STR_COL{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_join(cudf::strings_column_view(strs), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, print_all);
  }

  // Zero repeat time.
  {}

  // Sliced the first half of the column
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results     = cudf::strings::repeat_join(cudf::strings_column_view(sliced_strs), 2);
    auto const expected    = STR_COL{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }

  // Sliced the middle of the column
  {
    auto const sliced_strs = cudf::slice(strs, {1, 3})[0];
    auto const results     = cudf::strings::repeat_join(cudf::strings_column_view(sliced_strs), 2);
    auto const expected    = STR_COL{"abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, print_all);
  }
}

TEST_F(RepeatJoinStringTest, StringsColumnWithNulls)
{
  {

  }

  // Repeat once.
  {

  }

  // Zero repeat time.
  {}

  // Sliced the first half of the column
  {

  }

  // Sliced the middle of the column
  {
  }
}
