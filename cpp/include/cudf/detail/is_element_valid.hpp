/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Return validity of a row
 *
 * Retrieves the validity (NULL or non-NULL) of the specified row from device memory.
 *
 * @note Synchronizes `stream`.
 *
 * @throw cudf::logic_error if `element_index < 0 or >= col_view.size()`
 *
 * @param col_view The column to retrieve the validity from.
 * @param element_index The index of the row to retrieve.
 * @param stream The stream to use for copying the validity to the host.
 * @return Host boolean that indicates the validity of the row.
 */

bool is_element_valid_sync(column_view const& col_view,
                           size_type element_index,
                           rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
