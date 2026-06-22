/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>

#include <string_view>

/** @brief Target substring embedded in skewed string benchmark data. */
extern std::string_view const skewed_string_target_substring;

/**
 * @brief Deterministically generates a string column with a skewed length distribution.
 *
 * The column is built from 10 template strings of length 256 bytes.
 *
 * Rows are then assigned one of two lengths:
 * - `short_string_pct` percent of rows are cropped to the prefix `str[0:short_length]`
 * - The remaining rows repeat the template until `long_tail_length` and use
 * `str[0:long_tail_length]`
 *
 * Within the column, `hit_rate` percent of rows contain the target substring used by the
 * find/contains string benchmarks.
 *
 * @param num_rows Number of rows in the output column
 * @param short_length Length of the cropped short strings
 * @param long_tail_length Length of the repeated long tail rows
 * @param short_string_pct Percentage of rows that are short strings (0-100)
 * @param hit_rate Percentage of rows that contain the target substring (0-100)
 */
std::unique_ptr<cudf::column> create_skewed_string_column(cudf::size_type num_rows,
                                                          cudf::size_type short_length,
                                                          cudf::size_type long_tail_length,
                                                          int32_t short_string_pct,
                                                          int32_t hit_rate);
