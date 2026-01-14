/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_scan_util.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> min_scan(column_view const& values,
                                 size_type num_groups,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  return type_dispatcher(values.type(),
                         group_scan_dispatcher<aggregation::MIN>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
