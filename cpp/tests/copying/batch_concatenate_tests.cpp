/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <numeric>
#include <vector>

/**
 * @brief Test fixture for batch_concatenate tests.
 *
 * These tests focus on edge cases for the bitmask concatenation kernel,
 * particularly around word boundaries (32-bit alignment).
 */
struct BatchConcatenateTest : public cudf::test::BaseFixture {};

// Helper to create a column with specific validity pattern
template <typename T>
cudf::test::fixed_width_column_wrapper<T> make_column_with_validity(
  std::vector<T> const& values, std::vector<bool> const& validity)
{
  return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), validity.begin());
}

// ============================================================================
// Basic functionality tests
// ============================================================================

// ============================================================================
// Word boundary edge cases (32-bit alignment)
// ============================================================================

// ============================================================================
// Empty column edge cases
// ============================================================================

// ============================================================================
// Mixed null mask scenarios
// ============================================================================

// ============================================================================
// Source bit offset tests (sliced columns)
// ============================================================================

// ============================================================================
// Large column tests
// ============================================================================

// ============================================================================
// Specific bit pattern tests
// ============================================================================

// ============================================================================
// Struct column tests
// ============================================================================

TEST_F(BatchConcatenateTest, SmallPayloadFallsBackThroughPublicAPI)
{
  // An all-flat input under the 8 MiB payload threshold delegates to the standard
  // concatenate; this owns that decision for the column overload.
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 0, 1});
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}, {0, 1, 0});
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::batch_concatenate(columns);

  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6}, {1, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, EmptyTypedColumns)
{
  // All-EMPTY-typed sources take the early EMPTY path: sum the row counts, emit EMPTY output
  auto make_empty_rows = [](cudf::size_type rows) {
    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::EMPTY},
      rows,
      rmm::device_buffer{},
      cudf::create_null_mask(rows, cudf::mask_state::UNALLOCATED),
      rows);
  };
  auto c1 = make_empty_rows(3);
  auto c2 = make_empty_rows(2);
  std::vector<cudf::column_view> sources{c1->view(), c2->view()};

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  EXPECT_EQ(result->size(), 5);
  EXPECT_EQ(result->type().id(), cudf::type_id::EMPTY);
  EXPECT_EQ(result->null_count(), 5);
}

