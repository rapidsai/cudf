/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cmath>

auto constexpr null{0};  // null at current level
auto constexpr XXX{0};   // null pushed down from parent level
auto constexpr NaN          = std::numeric_limits<double>::quiet_NaN();
auto constexpr KEEP_ANY     = cudf::duplicate_keep_option::KEEP_ANY;
auto constexpr KEEP_FIRST   = cudf::duplicate_keep_option::KEEP_FIRST;
auto constexpr KEEP_LAST    = cudf::duplicate_keep_option::KEEP_LAST;
auto constexpr KEEP_NONE    = cudf::duplicate_keep_option::KEEP_NONE;
auto constexpr NULL_EQUAL   = cudf::null_equality::EQUAL;
auto constexpr NULL_UNEQUAL = cudf::null_equality::UNEQUAL;
auto constexpr NAN_EQUAL    = cudf::nan_equality::ALL_EQUAL;
auto constexpr NAN_UNEQUAL  = cudf::nan_equality::UNEQUAL;

using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col = cudf::test::fixed_width_column_wrapper<float>;

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

struct StableDistinctKeepAny : public cudf::test::BaseFixture {};

struct StableDistinctKeepFirstLastNone : public cudf::test::BaseFixture {};

TEST_F(StableDistinctKeepAny, NoNullsTableWithNaNs)
{
  // Column(s) used to test KEEP_ANY needs to have same rows in contiguous
  // groups for equivalent keys because KEEP_ANY is nondeterministic.
  auto const col1  = int32s_col{6, 6, 6, 1, 1, 1, 3, 5, 8, 5};
  auto const col2  = floats_col{6, 6, 6, 1, 1, 1, 3, 4, 9, 4};
  auto const keys1 = int32s_col{20, 20, 20, 15, 15, 15, 20, 19, 21, 9};
  auto const keys2 = floats_col{19., 19., 19., NaN, NaN, NaN, 20., 20., 9., 21.};

  auto const input   = cudf::table_view{{col1, col2, keys1, keys2}};
  auto const key_idx = std::vector<cudf::size_type>{2, 3};

  // NaNs are unequal.
  {
    auto const exp_col1  = int32s_col{6, 1, 1, 1, 3, 5, 8, 5};
    auto const exp_col2  = floats_col{6, 1, 1, 1, 3, 4, 9, 4};
    auto const exp_keys1 = int32s_col{20, 15, 15, 15, 20, 19, 21, 9};
    auto const exp_keys2 = floats_col{19., NaN, NaN, NaN, 20., 20., 9., 21.};
    auto const expected  = cudf::table_view{{exp_col1, exp_col2, exp_keys1, exp_keys2}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_EQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // NaNs are equal.
  {
    auto const exp_col1  = int32s_col{6, 1, 3, 5, 8, 5};
    auto const exp_col2  = floats_col{6, 1, 3, 4, 9, 4};
    auto const exp_keys1 = int32s_col{20, 15, 20, 19, 21, 9};
    auto const exp_keys2 = floats_col{19., NaN, 20., 20., 9., 21.};
    auto const expected  = cudf::table_view{{exp_col1, exp_col2, exp_keys1, exp_keys2}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_EQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(StableDistinctKeepAny, InputWithNullsAndNaNs)
{
  auto constexpr null{0.0};  // shadow the global `null` variable of type int

  // Column(s) used to test KEEP_ANY needs to have same rows in contiguous
  // groups for equivalent keys because KEEP_ANY is nondeterministic.
  auto const col   = int32s_col{5, 4, 4, 1, 1, 1, 8, 8, 1};
  auto const keys  = floats_col{{20., null, null, NaN, NaN, NaN, 19., 19., 21.}, nulls_at({1, 2})};
  auto const input = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // Nulls are equal, NaNs are unequal.
  {
    auto const exp_col  = int32s_col{5, 4, 1, 1, 1, 8, 1};
    auto const exp_keys = floats_col{{20., null, NaN, NaN, NaN, 19., 21.}, null_at(1)};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_EQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // Nulls are equal, NaNs are equal.
  {
    auto const exp_col  = int32s_col{5, 4, 1, 8, 1};
    auto const exp_keys = floats_col{{20., null, NaN, 19., 21.}, null_at(1)};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_EQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // Nulls are unequal, NaNs are unequal.
  {
    auto const exp_col  = int32s_col{5, 4, 4, 1, 1, 1, 8, 1};
    auto const exp_keys = floats_col{{20., null, null, NaN, NaN, NaN, 19., 21.}, nulls_at({1, 2})};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_UNEQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // Nulls are unequal, NaNs are equal.
  {
    auto const exp_col  = int32s_col{5, 4, 4, 1, 8, 1};
    auto const exp_keys = floats_col{{20., null, null, NaN, 19., 21.}, nulls_at({1, 2})};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_ANY, NULL_UNEQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(StableDistinctKeepFirstLastNone, InputWithNaNsEqual)
{
  // Column(s) used to test needs to have different rows for the same keys.
  auto const col     = int32s_col{0, 1, 2, 3, 4, 5, 6};
  auto const keys    = floats_col{20., NaN, NaN, 19., 21., 19., 22.};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // KEEP_FIRST
  {
    auto const exp_col  = int32s_col{0, 1, 3, 4, 6};
    auto const exp_keys = floats_col{20., NaN, 19., 21., 22.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_FIRST, NULL_EQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP_LAST
  {
    auto const exp_col  = int32s_col{0, 2, 4, 5, 6};
    auto const exp_keys = floats_col{20., NaN, 21., 19., 22.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_LAST, NULL_EQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP_NONE
  {
    auto const exp_col  = int32s_col{0, 4, 6};
    auto const exp_keys = floats_col{20., 21., 22.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_NONE, NULL_EQUAL, NAN_EQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(StableDistinctKeepFirstLastNone, InputWithNaNsUnequal)
{
  // Column(s) used to test needs to have different rows for the same keys.
  auto const col     = int32s_col{0, 1, 2, 3, 4, 5, 6, 7};
  auto const keys    = floats_col{20., NaN, NaN, 19., 21., 19., 22., 20.};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // KEEP_FIRST
  {
    auto const exp_col  = int32s_col{0, 1, 2, 3, 4, 6};
    auto const exp_keys = floats_col{20., NaN, NaN, 19., 21., 22.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_FIRST, NULL_UNEQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP_LAST
  {
    auto const exp_col  = int32s_col{1, 2, 4, 5, 6, 7};
    auto const exp_keys = floats_col{NaN, NaN, 21., 19., 22., 20.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_LAST, NULL_UNEQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP_NONE
  {
    auto const exp_col  = int32s_col{1, 2, 4, 6};
    auto const exp_keys = floats_col{NaN, NaN, 21., 22.};
    auto const expected = cudf::table_view{{exp_col, exp_keys}};

    auto const result = cudf::stable_distinct(
      input, key_idx, KEEP_NONE, NULL_UNEQUAL, NAN_UNEQUAL, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}
