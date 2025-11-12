/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/column/column_view.hpp>

namespace cudf::test {
class LargeStringsData;

/**
 * @brief Fixture for creating large strings tests
 *
 * Stores tests strings columns for reuse by specific tests.
 * Creating the test input only once helps speed up the overall tests.
 *
 * Also automatically enables appropriate large strings environment variables.
 */
struct StringsLargeTest : public cudf::test::BaseFixture {
  /**
   * @brief Returns a column of long strings
   *
   * This returns 8 rows of 400 bytes
   */
  cudf::column_view wide_column();

  /**
   * @brief Returns a long column of strings
   *
   * This returns 5 million rows of 50 bytes
   */
  cudf::column_view long_column();

  /**
   * @brief Returns a very long column of strings
   *
   * This returns 30 million rows of 5 bytes
   */
  cudf::column_view very_long_column();

  large_strings_enabler g_ls_enabler;
  static LargeStringsData* g_ls_data;

  static std::unique_ptr<LargeStringsData> get_ls_data();
};
}  // namespace cudf::test
