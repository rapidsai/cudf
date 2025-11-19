/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/merge.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace strings::detail {
/**
 * @brief Merges two strings columns
 *
 * @param lhs First column
 * @param rhs Second column
 * @param row_order Indices for each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace CUDF_EXPORT cudf
