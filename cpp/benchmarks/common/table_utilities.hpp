/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>

/**
 * @brief Estimates the column size in bytes.
 *
 * @remark As this function internally uses cudf::row_bit_count() to estimate each row size
 * and accumulates them, the returned estimate may be an inexact approximation in some
 * cases. See cudf::row_bit_count() for more details.
 *
 * @param view The column view to estimate its size
 */
int64_t estimate_size(cudf::column_view const& view);

/**
 * @brief Estimates the table size in bytes.
 *
 * @remark As this function internally uses cudf::row_bit_count() to estimate each row size
 * and accumulates them, the returned estimate may be an inexact approximation in some
 * cases. See cudf::row_bit_count() for more details.
 *
 * @param view The table view to estimate its size
 */
int64_t estimate_size(cudf::table_view const& view);
