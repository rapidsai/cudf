/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::sort_lists
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::cuda_stream_view stream,
                                   cudf::memory_resources resources);

/**
 * @copydoc cudf::lists::stable_sort_lists
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::cuda_stream_view stream,
                                          cudf::memory_resources resources);

}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
