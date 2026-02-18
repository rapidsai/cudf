/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::distinct_count(column_view const&, null_policy, nan_policy, rmm::cuda_stream_view)
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::distinct_count(table_view const&, null_equality, rmm::cuda_stream_view)
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
