/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/merge.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace dictionary::detail {

/**
 * @brief Merges two dictionary columns.
 *
 * The keys of both dictionary columns are expected to be already matched.
 * Otherwise, the result is undefined behavior.
 *
 * Caller must set the validity mask in the output column.
 *
 * @param lcol First column.
 * @param rcol Second column.
 * @param row_order Indexes for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> merge(dictionary_column_view const& lcol,
                              dictionary_column_view const& rcol,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace dictionary::detail
}  // namespace CUDF_EXPORT cudf
