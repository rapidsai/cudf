/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_top_k(size_type k,
                                    order topk_order,
                                    column_view const& values,
                                    cudf::device_span<size_type const> group_offsets,
                                    rmm::cuda_stream_view stream,
                                    cudf::memory_resources resources)
{
  return cudf::detail::segmented_top_k(values, group_offsets, k, topk_order, stream, resources);
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
