/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// These tests need several GiB of device memory, so they are in LARGE_STRINGS_TEST, which runs
// alone on a GPU, and they are skipped when the GPU does not have enough free memory.
struct ListsLevelsTest : public cudf::test::BaseFixture {
  /**
   * @brief Makes a lists column whose first rows hold `values`, `row_size` values per row, followed
   * by `num_empty` empty rows.
   */
  static std::unique_ptr<cudf::column> make_lists(std::unique_ptr<cudf::column> values,
                                                  cudf::size_type row_size,
                                                  cudf::size_type num_empty)
  {
    auto const num_values = values->size();
    std::vector<cudf::size_type> offsets;
    for (int64_t offset = 0; offset < num_values; offset += row_size) {
      offsets.push_back(static_cast<cudf::size_type>(offset));
    }
    offsets.insert(offsets.end(), num_empty + 1, num_values);
    auto const num_rows = static_cast<cudf::size_type>(offsets.size() - 1);
    auto offsets_column =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
        .release();
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_column),
                                   std::move(values),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  }

  /**
   * @brief Expects sorting and writing to Parquet, which compute the Dremel encoding of the column
   * with `get_comparator_data` and `get_dremel_data`, to throw `std::overflow_error`, and a
   * comparison of the column without its last empty list, which has exactly INT_MAX level values,
   * to succeed.
   */
  static void expect_overflow(cudf::column_view const& lists)
  {
    auto const input = cudf::table_view{{lists}};
    EXPECT_THROW(cudf::sorted_order(input), std::overflow_error);
    std::vector<char> buffer;
    EXPECT_THROW(
      cudf::io::write_parquet(
        cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, input).build()),
      std::overflow_error);

    auto const fits = cudf::slice(lists, {0, lists.size() - 1}).front();
    EXPECT_NO_THROW(cudf::is_sorted(cudf::table_view{{fits}}, {}, {}));
  }
};

// A list column has one level value per leaf value and one per empty or null list, so INT_MAX - 1
// values and two empty lists make one more level value than a column can hold.
auto constexpr num_values = std::numeric_limits<cudf::size_type>::max() - 1;
// Rows of 1M values, so that no row comes near the writer's limit for a single row
auto constexpr row_size = cudf::size_type{1} << 20;

TEST_F(ListsLevelsTest, TooManyLevelValues)
{
  // Racecheck takes over half an hour for the kernels over 2^31 values
  if (std::getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }
  // The values and the repetition and definition levels take about 6 GiB at the peak.
  auto constexpr required_bytes = std::size_t{7} << 30;
  if (rmm::available_device_memory().first < required_bytes) {
    GTEST_SKIP() << "Needs " << (required_bytes >> 30) << " GiB of free device memory";
  }

  auto values = cudf::make_column_from_scalar(cudf::numeric_scalar<int8_t>(0), num_values);
  auto lists  = make_lists(std::move(values), row_size, 2);
  expect_overflow(lists->view());
}

TEST_F(ListsLevelsTest, TooManyNestedLevelValues)
{
  // Racecheck takes over half an hour for the kernels over 2^31 values
  if (std::getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }
  // The temporary levels for merging the nested levels add about 4 GiB to the peak.
  auto constexpr required_bytes = std::size_t{12} << 30;
  if (rmm::available_device_memory().first < required_bytes) {
    GTEST_SKIP() << "Needs " << (required_bytes >> 30) << " GiB of free device memory";
  }

  // The empty lists are at the outer level, so the overflow happens when merging that level.
  auto values = cudf::make_column_from_scalar(cudf::numeric_scalar<int8_t>(0), num_values);
  auto inner  = make_lists(std::move(values), row_size, 0);
  auto lists  = make_lists(std::move(inner), 1, 2);
  expect_overflow(lists->view());
}
