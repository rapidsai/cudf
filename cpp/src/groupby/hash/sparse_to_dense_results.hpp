/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::groupby::detail::hash {
/**
 * @brief Gather sparse aggregation results into dense using `gather_map` and add to
 * `dense_results`
 *
 * @tparam SetRef Device hash set ref type
 *
 * @param[in] requests The set of columns to aggregate and the aggregations to perform
 * @param[in] sparse_results Sparse aggregation results
 * @param[out] dense_results Dense aggregation results
 * @param[in] gather_map Gather map indicating valid elements in `sparse_results`
 * @param[in] set Device hash set ref
 * @param[in] row_bitmask Bitmask indicating the validity of input keys
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table
 */
template <typename SetRef>
void sparse_to_dense_results(host_span<aggregation_request const> requests,
                             cudf::detail::result_cache* sparse_results,
                             cudf::detail::result_cache* dense_results,
                             device_span<size_type const> gather_map,
                             SetRef set,
                             bitmask_type const* row_bitmask,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
