/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distinct_helpers.cuh"

namespace cudf::detail {

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
