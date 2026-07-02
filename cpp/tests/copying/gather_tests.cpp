/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <numeric>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GatherTest, cudf::test::NumericTypes);

TYPED_TEST(GatherTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullOdds)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather odd-valued indices
  auto map_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data  = cuda::constant_iterator{0};
  auto expect_valid = cudf::test::iterators::all_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullEvens)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather even-valued indices
  auto map_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });
  auto expect_valid = cudf::test::iterators::no_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};

  // Every element is invalid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::all_nulls();

  // Create a gather map that gathers to random locations
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_map_data.begin(), host_map_data.end(), g);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{
    data, data + source_size, validity};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(host_map_data.begin(),
                                                             host_map_data.end());

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Check that the result is also all invalid
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, MultiColReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  constexpr cudf::size_type n_cols = 3;

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, MultiColNulls)
{
  constexpr cudf::size_type source_size{1000};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");

  constexpr cudf::size_type n_cols = 3;

  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size, validity));
    source_columns.push_back(source_column_wrappers[i]);
  }

  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Expected data
  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - i - 1; });
  auto expect_valid = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

// ---------------------------------------------------------------------------
// Stream-pool fork/join path tests.
// The fork threshold is min_bytes_for_stream_fork = 512 * 1024 bytes.
// For int64 (8 bytes), the threshold row count is 65536.
// For int32 (4 bytes), the threshold row count is 131072.
// ---------------------------------------------------------------------------

class GatherStreamPoolTest : public cudf::test::BaseFixture {};

// Below threshold: 2 int64 columns × 32768 rows = 256 KB < 512 KB → single-stream path
TEST_F(GatherStreamPoolTest, BelowThreshold_SingleStreamPath)
{
  constexpr cudf::size_type num_rows = 32'768;
  constexpr int n_cols               = 2;

  auto data = cuda::counting_iterator{0};
  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + num_rows);
    col_views.push_back(cols.back());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + num_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

// At threshold: 4 int64 columns × 65536 rows = 512 KB/col → fork/join path (half-cap tier)
TEST_F(GatherStreamPoolTest, AtThreshold_ForkJoinHalfCap)
{
  constexpr cudf::size_type num_rows = 65'536;
  constexpr int n_cols               = 4;

  auto data = cuda::counting_iterator{0};
  auto reversed =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return num_rows - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + num_rows);
    col_views.push_back(cols.back());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed, reversed + num_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map);

  cudf::test::fixed_width_column_wrapper<int64_t> expected(reversed, reversed + num_rows);
  for (int c = 0; c < n_cols; ++c) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view().column(c));
  }
}

// Full-cap tier: 9 int64 columns × 1100000 rows = 8.8 MB >= 8 MB → 8 streams
// 9 columns > 8 streams exercises round-robin assignment (i % num_streams)
TEST_F(GatherStreamPoolTest, FullCapTier_RoundRobin)
{
  constexpr cudf::size_type num_rows = 1'100'000;
  constexpr int n_cols               = 9;

  auto data = cuda::counting_iterator{0};
  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + num_rows);
    col_views.push_back(cols.back());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + num_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

// Single column does NOT fork regardless of size
TEST_F(GatherStreamPoolTest, SingleColumnNoFork)
{
  constexpr cudf::size_type num_rows = 70'000;

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<int64_t> col(data, data + num_rows);
  cudf::table_view source_table{{col}};

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + num_rows);
  auto result = cudf::gather(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

// Fork path with nullable columns — exercises gather_bitmask on joined stream
TEST_F(GatherStreamPoolTest, NullableMultiCol_ForkPath)
{
  constexpr cudf::size_type num_rows = 70'000;
  constexpr int n_cols               = 2;

  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto reversed =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return num_rows - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + num_rows, validity);
    col_views.push_back(cols.back());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed, reversed + num_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map);

  // After reversing, row i came from row (num_rows-1-i). That row is valid iff (num_rows-1-i) % 2.
  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return num_rows - 1 - i; });
  auto expect_valid = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (num_rows - 1 - i) % 2; });
  cudf::test::fixed_width_column_wrapper<int64_t> expected(
    expect_data, expect_data + num_rows, expect_valid);

  for (int c = 0; c < n_cols; ++c) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view().column(c));
  }
}

// NULLIFY policy with out-of-bounds indices on fork path.
// int64 columns: bytes_per_col = 65536 * 8 = 512 KB >= 512 KB threshold → fork path taken.
TEST_F(GatherStreamPoolTest, NullifyOOB_ForkPath)
{
  constexpr cudf::size_type source_rows = 131'073;  // Must exceed gather_rows.
  constexpr cudf::size_type gather_rows = 65'536;   // Exactly at 512 KB threshold for int64.
  constexpr int n_cols                  = 2;

  auto data = cuda::counting_iterator<int64_t>{0};
  // Alternate valid/OOB indices to trigger NULLIFY.
  auto oob_map = cudf::detail::make_counting_transform_iterator(
    0, [source_rows](auto i) { return i % 2 == 0 ? i : source_rows + 1; });

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + source_rows);
    col_views.push_back(cols.back());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(oob_map, oob_map + gather_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY);
  EXPECT_EQ(result->num_columns(), n_cols);

  // Even positions gather source[i] = i; odd positions are OOB → null.
  auto expect_data = cuda::counting_iterator<int64_t>{0};
  auto expect_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? 1 : 0; });
  cudf::test::fixed_width_column_wrapper<int64_t> expected(
    expect_data, expect_data + gather_rows, expect_valid);

  for (int c = 0; c < n_cols; ++c) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view().column(c));
  }
}

// Narrow type (int8): 1 byte × 524288 rows = 512 KB = threshold.
// Exercises max_elem_bytes=1 path and verifies fork/join for narrow types.
TEST_F(GatherStreamPoolTest, NarrowType_AtThreshold)
{
  constexpr cudf::size_type num_rows = 524'288;
  constexpr int n_cols               = 2;

  auto data = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<int8_t>(i % 128); });
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> col_views;
  for (int c = 0; c < n_cols; ++c) {
    cols.emplace_back(data, data + num_rows);
    col_views.push_back(cols.back());
  }
  auto map_data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data, map_data + num_rows);
  cudf::table_view source_table{col_views};

  auto result = cudf::gather(source_table, gather_map);
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}
